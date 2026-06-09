"""
Generative analog design via two complementary strategies:

1. **SMARTS reaction transforms** — a curated library of common medicinal
   chemistry moves (halogen scan, methyl/CF3/OMe addition, bioisostere
   replacements) applied directly to the query molecule.

2. **MMP fragmentation + R-group recombination** — the query is cut at every
   rotatable single bond using rdMMPA; each resulting (core, R-group) pair is
   recombined with a small set of drug-like fragment replacements.

All generated analogs are de-duplicated, scored by predicted property,
Tanimoto similarity to the query, and synthetic accessibility, then ranked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts

from mol_pilot._sa_score import sa_score
from mol_pilot.interpret import tanimoto_similarity

logger = logging.getLogger(__name__)


# ── MedChem SMARTS transformation library ────────────────────────────────────
#
# Format: (name, reaction_smarts)
# The reaction SMARTS must be well-defined and terminate quickly on drug-like mols.

_TRANSFORMS: list[tuple[str, str]] = [
    # Halogen scan — aromatic H → halogen
    ("ArH→ArF",    "[c:1][H]>>[c:1]F"),
    ("ArH→ArCl",   "[c:1][H]>>[c:1]Cl"),
    ("ArH→ArBr",   "[c:1][H]>>[c:1]Br"),
    # Methyl scan
    ("ArH→ArMe",   "[c:1][H]>>[c:1]C"),
    ("ArH→ArOMe",  "[c:1][H]>>[c:1]OC"),
    ("ArH→ArCF3",  "[c:1][H]>>[c:1]C(F)(F)F"),
    ("ArH→ArCN",   "[c:1][H]>>[c:1]C#N"),
    # Sp3 C–H fluorination
    ("AlkH→AlkF",  "[CX4:1][H]>>[CX4:1]F"),
    # N-methylation
    ("NH→NMe",     "[N;H1;!$(N-C=O):1]>>[N:1]C"),
    # O-methylation
    ("OH→OMe",     "[O;H1:1]>>[O:1]C"),
    # Hydroxyl ↔ fluoro bioisostere
    ("OH→F",       "[O;H1:1]>>[F:1]"),
    # COOH → amide (simple)
    ("COOH→CONH2", "[C:1](=[O:2])[OH]>>[C:1](=[O:2])N"),
    # NH2 → NHMe
    ("NH2→NHMe",   "[N;H2:1]>>[N;H1:1]C"),
    # Methyl homologation
    ("Me→Et",      "[c:1][CH3:2]>>[c:1][CH2:2]C"),
    # Remove methyl
    ("Me→H_Ar",    "[c:1][CH3]>>[c:1][H]"),
    # N-oxide formation (important for solubility)
    ("Py→PyNOx",   "[n:1]>>[n+:1][O-]"),
]


def _build_reactions() -> list[tuple[str, ChemicalReaction]]:
    valid = []
    for name, smarts in _TRANSFORMS:
        try:
            rxn = ReactionFromSmarts(smarts)
            if rxn is not None:
                valid.append((name, rxn))
        except Exception as exc:
            logger.debug("Could not compile reaction %s: %s", name, exc)
    return valid


_REACTIONS: list[tuple[str, ChemicalReaction]] = _build_reactions()


# ── Fragment R-group library for MMP recombination ────────────────────────────

_RGROUP_SMILES: list[str] = [
    "[*:1]F",
    "[*:1]Cl",
    "[*:1]Br",
    "[*:1]C",
    "[*:1]CC",
    "[*:1]OC",
    "[*:1]C(F)(F)F",
    "[*:1]C#N",
    "[*:1]C(=O)N",
    "[*:1]C(=O)O",
    "[*:1]N",
    "[*:1]NC",
    "[*:1]NC(=O)C",
    "[*:1]c1ccccc1",
    "[*:1]c1ccncc1",
    "[*:1]S(=O)(=O)N",
    "[*:1]C1CC1",
    "[*:1]C1CCNCC1",
]


# ── Core scoring helpers ──────────────────────────────────────────────────────

def _ecfp4(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _tanimoto(fp_q: DataStructs.ExplicitBitVect, mol: Chem.Mol) -> float:
    fp = _ecfp4(mol)
    return float(DataStructs.TanimotoSimilarity(fp_q, fp))


def _lipinski_ok(mol: Chem.Mol) -> bool:
    mw = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)
    return mw <= 600 and hbd <= 5 and hba <= 10 and logp <= 6.0


# ── Transform-based analog generation ────────────────────────────────────────

def _apply_transforms(mol: Chem.Mol) -> list[tuple[str, Chem.Mol]]:
    """Apply all SMARTS transforms; return (transform_name, product_mol) pairs."""
    results: list[tuple[str, Chem.Mol]] = []
    for name, rxn in _REACTIONS:
        try:
            products = rxn.RunReactants((mol,))
        except Exception:
            continue
        for prod_tuple in products[:3]:  # take at most 3 products per transform
            prod = prod_tuple[0]
            try:
                Chem.SanitizeMol(prod)
                smi = Chem.MolToSmiles(prod, canonical=True)
                if smi and Chem.MolFromSmiles(smi) is not None:
                    results.append((name, prod))
            except Exception:
                continue
    return results


# ── MMP-based analog generation ──────────────────────────────────────────────

def _mmp_analogs(mol: Chem.Mol) -> list[tuple[str, Chem.Mol]]:
    """Fragment molecule via rdMMPA and swap R-groups with the library."""
    try:
        from rdkit.Chem import rdMMPA
    except ImportError:
        return []

    results: list[tuple[str, Chem.Mol]] = []
    try:
        frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
    except Exception as exc:
        logger.debug("rdMMPA fragmentation failed: %s", exc)
        return results

    for core_smi, rg_smi in frags:
        if core_smi is None or "[*:1]" not in core_smi:
            continue
        core_mol = Chem.MolFromSmiles(core_smi)
        if core_mol is None:
            continue

        for new_rg in _RGROUP_SMILES:
            if new_rg == rg_smi:
                continue  # skip identical R-group
            # Replace [*:1] in core with [*:1] attachment from library
            try:
                rxn_sma = f"[*:1]>>{new_rg.replace('[*:1]', '[*:1]')}"
                # Build full SMILES by simple substitution
                new_smi = core_smi.replace("[*:1]", new_rg.replace("[*:1]", ""))
                # Clean up: remove any remaining dummy atoms
                new_smi = new_smi.replace("[*:1]", "").replace("[*]", "")
                candidate = Chem.MolFromSmiles(new_smi)
                if candidate is not None:
                    Chem.SanitizeMol(candidate)
                    results.append((f"MMP({new_rg})", candidate))
            except Exception:
                continue

    return results


# ── Main public interface ─────────────────────────────────────────────────────

@dataclass
class AnalogSuggestion:
    smiles: str
    transform: str
    predicted_property: float
    delta_property: float
    tanimoto: float
    sa_score: float
    mw: float
    logp: float
    passes_lipinski: bool
    rank: int = 0


def generate_analogs(
    smiles: str,
    predictor: Any,
    n_analogs: int = 25,
    include_mmp: bool = True,
    min_tanimoto: float = 0.30,
    max_tanimoto: float = 0.99,
    max_sa: float = 7.0,
    optimize_direction: str = "maximize",
) -> pd.DataFrame:
    """Generate, score, and rank analog suggestions for a query molecule.

    Parameters
    ----------
    smiles:
        Query molecule SMILES.
    predictor:
        Any object with a ``predict(list[str]) -> ndarray`` method.
    n_analogs:
        Maximum analogs to return.
    include_mmp:
        Include rdMMPA-based fragment-swap analogs (slower but more diverse).
    min_tanimoto / max_tanimoto:
        Similarity band to the query (filters near-identical and unrelated).
    max_sa:
        Maximum SA score to allow (prunes very hard-to-synthesise structures).
    optimize_direction:
        ``"maximize"`` (higher property = better) or ``"minimize"``.
    """
    query_mol = Chem.MolFromSmiles(smiles)
    if query_mol is None:
        raise ValueError(f"Invalid query SMILES: {smiles!r}")

    query_pred = float(predictor.predict([smiles])[0])
    query_fp = _ecfp4(query_mol)

    # Generate candidates
    candidates: list[tuple[str, str, Chem.Mol]] = []
    for name, mol in _apply_transforms(query_mol):
        smi = Chem.MolToSmiles(mol, canonical=True)
        candidates.append((smi, name, mol))

    if include_mmp:
        for name, mol in _mmp_analogs(query_mol):
            smi = Chem.MolToSmiles(mol, canonical=True)
            candidates.append((smi, name, mol))

    # De-duplicate by canonical SMILES (and remove query itself)
    seen: set[str] = {Chem.MolToSmiles(query_mol, canonical=True)}
    unique: list[tuple[str, str, Chem.Mol]] = []
    for smi, name, mol in candidates:
        if smi not in seen:
            seen.add(smi)
            unique.append((smi, name, mol))

    if not unique:
        logger.warning("No analogs generated for %s", smiles)
        return pd.DataFrame()

    # Score all unique analogs
    all_smiles = [s for s, _, _ in unique]
    preds = predictor.predict(all_smiles)

    rows: list[AnalogSuggestion] = []
    for (smi, name, mol), pred in zip(unique, preds):
        sim = float(DataStructs.TanimotoSimilarity(query_fp, _ecfp4(mol)))
        if sim < min_tanimoto or sim > max_tanimoto:
            continue
        sas = sa_score(mol)
        if sas > max_sa:
            continue

        delta = float(pred) - query_pred
        rows.append(
            AnalogSuggestion(
                smiles=smi,
                transform=name,
                predicted_property=float(pred),
                delta_property=delta,
                tanimoto=sim,
                sa_score=sas,
                mw=Descriptors.MolWt(mol),
                logp=Descriptors.MolLogP(mol),
                passes_lipinski=_lipinski_ok(mol),
            )
        )

    if not rows:
        return pd.DataFrame()

    # Rank by composite score: property improvement + similarity + ease of synthesis
    sign = 1.0 if optimize_direction == "maximize" else -1.0
    for r in rows:
        r._score = sign * r.delta_property + 0.5 * r.tanimoto - 0.1 * r.sa_score

    rows.sort(key=lambda r: r._score, reverse=True)  # type: ignore[attr-defined]
    for rank, r in enumerate(rows[:n_analogs], 1):
        r.rank = rank

    df = pd.DataFrame(
        [
            {
                "rank": r.rank,
                "smiles": r.smiles,
                "transform": r.transform,
                "predicted_property": round(r.predicted_property, 3),
                "delta_property": round(r.delta_property, 3),
                "tanimoto_to_query": round(r.tanimoto, 3),
                "sa_score": round(r.sa_score, 2),
                "mol_weight": round(r.mw, 1),
                "logp": round(r.logp, 2),
                "passes_lipinski": r.passes_lipinski,
            }
            for r in rows[:n_analogs]
        ]
    )
    return df

"""
Synthetic accessibility (SA) score implementation.

Attempts to use the Ertl & Schuffenhauer (2009) SA score from RDKit's Contrib
directory.  If the fragment-frequency pickle is not found (e.g. minimal pip
install), falls back to a structural proxy that correlates well with the
original score for drug-like molecules (ring complexity + stereocentres +
macrocycles).

Scores are in [1, 10] where 1 = very easy to synthesise, 10 = very hard.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

_ertl_scorer = None
_ertl_attempted = False


def _try_load_ertl() -> bool:
    """Try once to import the Ertl SA scorer; cache result."""
    global _ertl_scorer, _ertl_attempted
    if _ertl_attempted:
        return _ertl_scorer is not None
    _ertl_attempted = True
    try:
        # RDKit installs the SA_Score module in rdkit.Contrib.SA_Score
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore[import]
        _ertl_scorer = sascorer
        logger.debug("Using Ertl SA scorer (rdkit.Contrib.SA_Score)")
        return True
    except Exception:
        pass
    try:
        import importlib, sys
        # Some RDKit builds expose it differently
        spec = importlib.util.find_spec("sascorer")
        if spec:
            sascorer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sascorer)
            _ertl_scorer = sascorer
            logger.debug("Using sascorer from path")
            return True
    except Exception:
        pass
    logger.debug("Ertl SA scorer not available; using structural proxy")
    return False


def _structural_proxy(mol: Chem.Mol) -> float:
    """Proxy SA score based on ring complexity and stereocentres.

    Calibrated so simple flat aromatic drugs score ~2-3 and complex natural
    products score ~7-9, in rough agreement with the Ertl score.
    """
    ring_info = mol.GetRingInfo()
    n_rings = len(ring_info.AtomRings())

    # Fused ring systems: pairs of rings sharing ≥ 2 atoms
    atom_rings = ring_info.AtomRings()
    n_fused = 0
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            if len(set(atom_rings[i]) & set(atom_rings[j])) >= 2:
                n_fused += 1

    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    n_stereo = len(chiral_centers)

    n_macro = sum(1 for r in ring_info.AtomRings() if len(r) > 8)
    n_heavy = mol.GetNumHeavyAtoms()

    score = (
        1.0
        + 0.25 * max(0, n_rings - 1)
        + 0.30 * n_fused
        + 0.40 * n_stereo
        + 0.80 * n_spiro
        + 0.70 * n_bridge
        + 2.00 * n_macro
        + 0.008 * max(0, n_heavy - 25)
    )

    # Small correction for highly heteroatom-rich molecules
    n_hetero = sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() not in (1, 6)
    )
    score += 0.04 * max(0, n_hetero - 4)

    return float(min(10.0, max(1.0, score)))


def sa_score(mol: Chem.Mol) -> float:
    """Return the synthetic accessibility score in [1, 10].

    Uses the Ertl scorer when available, proxy otherwise.
    """
    if mol is None:
        return 10.0
    if _try_load_ertl() and _ertl_scorer is not None:
        try:
            return float(_ertl_scorer.calculateScore(mol))
        except Exception as exc:
            logger.debug("Ertl scorer failed (%s); falling back to proxy", exc)
    return _structural_proxy(mol)


def sa_score_from_smiles(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    return sa_score(mol)

"""
Interpretability: SHAP attributions mapped to atoms and activity-cliff detection.

Key ideas
---------
- ECFP bits are Morgan environments anchored at specific atoms.  BitInfo returned
  by RDKit tells us which atoms contributed to each bit that is ON.  We accumulate
  the SHAP value of each bit onto its contributing atoms to produce per-atom
  importance weights that can be visualised as a coloured 2-D structure.
- Activity cliffs are (structurally similar, property-dissimilar) pairs — a
  signal that a small structural change flips a major property.  We flag pairs
  where Tanimoto similarity ≥ ``sim_threshold`` AND |Δproperty| ≥ ``delta_threshold``.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)


# ── atom-level SHAP weights ──────────────────────────────────────────────────

def get_bit_atom_map(
    mol: Chem.Mol,
    radius: int = 2,
    nbits: int = 2048,
    use_chirality: bool = False,
    use_features: bool = False,
) -> dict[int, list[int]]:
    """Return ``{bit_index: [atom_idx, ...]}`` for bits that are ON in *mol*."""
    bi: dict[int, tuple[tuple[int, int], ...]] = {}
    AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=nbits,
        useChirality=use_chirality,
        useFeatures=use_features,
        bitInfo=bi,
    )
    return {bit: list({a for a, _ in hits}) for bit, hits in bi.items()}


def map_shap_to_atoms(
    mol: Chem.Mol,
    shap_values: np.ndarray,
    radius: int = 2,
    nbits: int = 2048,
) -> np.ndarray:
    """Convert per-bit SHAP values to per-atom importance weights.

    Atoms touched by multiple bits accumulate contributions and are then
    averaged so that highly-connected atoms are not artificially up-weighted.

    Returns a float array of shape ``(n_atoms,)``.
    """
    bit_map = get_bit_atom_map(mol, radius=radius, nbits=nbits)
    atom_sum = np.zeros(mol.GetNumAtoms(), dtype=float)
    atom_cnt = np.zeros(mol.GetNumAtoms(), dtype=float)

    for bit_idx, sv in enumerate(shap_values):
        if bit_idx in bit_map:
            for atom_idx in bit_map[bit_idx]:
                atom_sum[atom_idx] += sv
                atom_cnt[atom_idx] += 1

    mask = atom_cnt > 0
    atom_weights = np.zeros_like(atom_sum)
    atom_weights[mask] = atom_sum[mask] / atom_cnt[mask]
    return atom_weights


# ── SHAP interpreter class ───────────────────────────────────────────────────

class SHAPInterpreter:
    """Compute and visualise SHAP attributions for an ECFP-based predictor.

    Parameters
    ----------
    predictor:
        A fitted ``LightGBMPredictor`` or ``SklearnPredictor``.
    background_smiles:
        A representative background set used to initialise the TreeExplainer.
        If None, the full training set is used (less efficient).
    """

    def __init__(self, predictor: Any, background_smiles: list[str] | None = None) -> None:
        self.predictor = predictor
        self._explainer: Any | None = None
        self._background_smiles = background_smiles

    def _build_explainer(self) -> None:
        import shap

        bg = self._background_smiles
        if bg is not None:
            X_bg = self.predictor.featurizer.transform(bg)
            # Use a small background summary for speed
            if len(X_bg) > 100:
                X_bg = shap.sample(X_bg, 100, random_state=42)
            self._explainer = shap.TreeExplainer(self.predictor.model, X_bg)
        else:
            self._explainer = shap.TreeExplainer(self.predictor.model)

    def explain(self, smiles: str) -> np.ndarray:
        """Return SHAP values (per fingerprint bit) for a single molecule."""
        if self._explainer is None:
            self._build_explainer()

        X = self.predictor.featurizer.transform([smiles])
        shap_vals = self._explainer.shap_values(X)
        # Handle multi-output (classification): take first class or mean
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        return np.asarray(shap_vals[0])

    def atom_weights(self, smiles: str) -> np.ndarray:
        """Map SHAP bit values back to atoms for a single molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        sv = self.explain(smiles)

        feat = self.predictor.featurizer
        radius = getattr(feat, "radius", 2)
        nbits = getattr(feat, "nbits", 2048)
        return map_shap_to_atoms(mol, sv, radius=radius, nbits=nbits)

    def draw_highlighted_mol(
        self,
        smiles: str,
        size: tuple[int, int] = (600, 400),
        positive_color: tuple[float, float, float] = (0.12, 0.72, 0.28),
        negative_color: tuple[float, float, float] = (0.85, 0.17, 0.17),
        return_svg: bool = False,
    ) -> Any:
        """Return a PIL image (or SVG string) of the molecule coloured by SHAP weights.

        Positive SHAP (activating for property) → green.
        Negative SHAP (suppressing for property) → red.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        weights = self.atom_weights(smiles)
        max_w = np.abs(weights).max() or 1.0
        norm_w = weights / max_w

        atom_colors: dict[int, tuple[float, float, float]] = {}
        for i, w in enumerate(norm_w):
            if w > 0:
                r, g, b = positive_color
                atom_colors[i] = (r * 0.3 + 0.7 * w * r, g * 0.3 + 0.7 * w * g, b)
            else:
                aw = abs(w)
                r, g, b = negative_color
                atom_colors[i] = (r, g * 0.3 + 0.7 * (1 - aw) * g, b * 0.3)

        highlight_atoms = list(range(mol.GetNumAtoms()))
        highlight_bonds: list[int] = []
        bond_colors: dict[int, tuple[float, float, float]] = {}

        if return_svg:
            drawer = rdMolDraw2D.MolDraw2DSVG(*size)
        else:
            drawer = rdMolDraw2D.MolDraw2DCairo(*size)

        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=highlight_bonds,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()

        if return_svg:
            return drawer.GetDrawingText()

        from PIL import Image
        bio = io.BytesIO(drawer.GetDrawingText())
        return Image.open(bio).copy()

    def top_features(self, smiles: str, n: int = 10) -> list[dict]:
        """Return the *n* fingerprint bits with the largest absolute SHAP values."""
        sv = self.explain(smiles)
        top_idx = np.argsort(np.abs(sv))[::-1][:n]
        return [
            {"bit": int(i), "shap_value": float(sv[i]), "direction": "activating" if sv[i] > 0 else "suppressing"}
            for i in top_idx
        ]


# ── activity-cliff detection ──────────────────────────────────────────────────

@dataclass
class ActivityCliff:
    smiles_a: str
    smiles_b: str
    tanimoto: float
    delta_property: float
    is_cliff: bool
    label: str


def tanimoto_similarity(smiles_a: str, smiles_b: str, radius: int = 2, nbits: int = 2048) -> float:
    """Compute Tanimoto similarity between two SMILES via ECFP4."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=nbits)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=nbits)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def activity_cliff_score(
    smiles_a: str,
    smiles_b: str,
    prop_a: float,
    prop_b: float,
    sim_threshold: float = 0.70,
    delta_threshold: float = 1.0,
    radius: int = 2,
    nbits: int = 2048,
) -> ActivityCliff:
    """Flag an (A, B) pair as an activity cliff.

    A cliff is defined as: Tanimoto ≥ *sim_threshold* AND |ΔP| ≥ *delta_threshold*.
    The defaults (0.70, 1.0 log unit) are widely used in SAR analysis.
    """
    sim = tanimoto_similarity(smiles_a, smiles_b, radius=radius, nbits=nbits)
    delta = abs(prop_a - prop_b)
    is_cliff = sim >= sim_threshold and delta >= delta_threshold

    if is_cliff:
        label = (
            f"⚠ Activity cliff: Tanimoto={sim:.2f} ≥ {sim_threshold}, "
            f"|ΔP|={delta:.2f} ≥ {delta_threshold}"
        )
    else:
        parts = []
        if sim < sim_threshold:
            parts.append(f"dissimilar (Tanimoto={sim:.2f} < {sim_threshold})")
        if delta < delta_threshold:
            parts.append(f"low property delta ({delta:.2f} < {delta_threshold})")
        label = "Not a cliff: " + "; ".join(parts)

    return ActivityCliff(
        smiles_a=smiles_a,
        smiles_b=smiles_b,
        tanimoto=sim,
        delta_property=delta,
        is_cliff=is_cliff,
        label=label,
    )


def screen_for_cliffs(
    smiles_list: list[str],
    properties: list[float],
    sim_threshold: float = 0.70,
    delta_threshold: float = 1.0,
) -> list[ActivityCliff]:
    """Screen all pairs in a compound list and return confirmed activity cliffs."""
    cliffs: list[ActivityCliff] = []
    n = len(smiles_list)
    for i in range(n):
        for j in range(i + 1, n):
            result = activity_cliff_score(
                smiles_list[i],
                smiles_list[j],
                properties[i],
                properties[j],
                sim_threshold=sim_threshold,
                delta_threshold=delta_threshold,
            )
            if result.is_cliff:
                cliffs.append(result)
    return cliffs

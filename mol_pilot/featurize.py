"""
Molecular featurization: ECFP fingerprints and RDKit 2D descriptors.

All featurizers implement the Featurizer protocol so they can be composed
or swapped via dependency injection without changing downstream model code.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

logger = logging.getLogger(__name__)


@runtime_checkable
class Featurizer(Protocol):
    """Pluggable featurizer interface — any object with these two methods qualifies."""

    def transform(self, smiles: list[str]) -> np.ndarray:
        """Return feature matrix of shape (n_molecules, n_features)."""
        ...

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        ...


# ── ECFP (Morgan) fingerprints ────────────────────────────────────────────────

class ECFPFeaturizer:
    """Extended connectivity fingerprint (Morgan) featurizer.

    Parameters
    ----------
    radius:
        Morgan radius (2 → ECFP4, 3 → ECFP6).
    nbits:
        Bit vector length.
    use_chirality:
        Include stereochemistry in the fingerprint.
    use_features:
        Use pharmacophoric feature invariants (FCFP mode).
    """

    def __init__(
        self,
        radius: int = 2,
        nbits: int = 2048,
        use_chirality: bool = False,
        use_features: bool = False,
    ) -> None:
        self.radius = radius
        self.nbits = nbits
        self.use_chirality = use_chirality
        self.use_features = use_features

    def transform(self, smiles: list[str]) -> np.ndarray:
        matrix = np.zeros((len(smiles), self.nbits), dtype=np.float32)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else smi
            if mol is None:
                logger.debug("Skipping invalid SMILES at index %d", i)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                self.radius,
                nBits=self.nbits,
                useChirality=self.use_chirality,
                useFeatures=self.use_features,
            )
            matrix[i] = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
        return matrix

    def get_feature_names(self) -> list[str]:
        return [f"ecfp_{i}" for i in range(self.nbits)]

    def get_bit_info(self, smiles: str) -> dict[int, list[tuple[int, int]]]:
        """Return {bit_index: [(atom_idx, radius), ...]} for a single molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        bi: dict[int, tuple[tuple[int, int], ...]] = {}
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            self.radius,
            nBits=self.nbits,
            useChirality=self.use_chirality,
            useFeatures=self.use_features,
            bitInfo=bi,
        )
        return {k: list(v) for k, v in bi.items()}

    def __repr__(self) -> str:
        return (
            f"ECFPFeaturizer(radius={self.radius}, nbits={self.nbits}, "
            f"chirality={self.use_chirality})"
        )


# ── RDKit 2D descriptors ──────────────────────────────────────────────────────

# Descriptors that are frequently NaN / infinity — exclude by default.
_UNSTABLE_DESCRIPTORS = {
    "Ipc",           # can overflow for large molecules
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
}

_ALL_DESCS: list[tuple[str, object]] = [
    (name, fn)
    for name, fn in Descriptors.descList
    if name not in _UNSTABLE_DESCRIPTORS
]


class RDKitDescriptorFeaturizer:
    """Compute RDKit 2D descriptors.

    Parameters
    ----------
    descriptor_names:
        Explicit list of descriptor names to use. Defaults to all stable
        RDKit descriptors.
    fill_na:
        Value to substitute for NaN/inf entries (default: 0.0).
    """

    def __init__(
        self,
        descriptor_names: list[str] | None = None,
        fill_na: float = 0.0,
    ) -> None:
        self.fill_na = fill_na
        if descriptor_names is None:
            self._descs = _ALL_DESCS
        else:
            name_set = set(descriptor_names)
            self._descs = [(n, fn) for n, fn in Descriptors.descList if n in name_set]

    def transform(self, smiles: list[str]) -> np.ndarray:
        n_desc = len(self._descs)
        matrix = np.full((len(smiles), n_desc), self.fill_na, dtype=np.float64)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else smi
            if mol is None:
                continue
            for j, (_, fn) in enumerate(self._descs):
                try:
                    val = fn(mol)
                    if val is not None and np.isfinite(val):
                        matrix[i, j] = val
                except Exception:
                    pass
        return matrix

    def get_feature_names(self) -> list[str]:
        return [name for name, _ in self._descs]

    def __repr__(self) -> str:
        return f"RDKitDescriptorFeaturizer(n_descriptors={len(self._descs)})"


# ── Combined featurizer ───────────────────────────────────────────────────────

class CombinedFeaturizer:
    """Concatenate outputs from two or more Featurizer instances."""

    def __init__(self, *featurizers: Featurizer) -> None:
        if len(featurizers) < 2:
            raise ValueError("Provide at least two featurizers to combine.")
        self.featurizers = featurizers

    def transform(self, smiles: list[str]) -> np.ndarray:
        parts = [f.transform(smiles) for f in self.featurizers]
        return np.hstack(parts)

    def get_feature_names(self) -> list[str]:
        names: list[str] = []
        for f in self.featurizers:
            names.extend(f.get_feature_names())
        return names

    def __repr__(self) -> str:
        inner = ", ".join(repr(f) for f in self.featurizers)
        return f"CombinedFeaturizer({inner})"

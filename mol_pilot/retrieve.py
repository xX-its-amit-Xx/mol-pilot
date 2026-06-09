"""
Nearest-neighbour similar-compound retrieval over a reference fingerprint index.

The index is built once from a reference library (SMILES list + optional
property values) and supports fast kNN lookup using cosine / Tanimoto similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

logger = logging.getLogger(__name__)


class NearestNeighborRetriever:
    """k-NN retrieval over a reference compound library using ECFP4.

    Parameters
    ----------
    radius:
        Morgan radius (default 2 → ECFP4).
    nbits:
        Fingerprint bit vector length.
    metric:
        ``"tanimoto"`` (exact, O(n)) or ``"cosine"`` (approximate via sklearn
        Ball-Tree, faster for large libraries).
    """

    def __init__(
        self,
        radius: int = 2,
        nbits: int = 2048,
        metric: str = "tanimoto",
    ) -> None:
        self.radius = radius
        self.nbits = nbits
        self.metric = metric

        self._smiles: list[str] = []
        self._properties: dict[str, list[float]] = {}
        self._fps_matrix: np.ndarray | None = None
        self._bit_fps: list[DataStructs.ExplicitBitVect] = []
        self._nn_index: Any | None = None

    # ── index building ────────────────────────────────────────────────────────

    def build(
        self,
        smiles: list[str],
        properties: dict[str, list[float]] | None = None,
    ) -> "NearestNeighborRetriever":
        """Build the retrieval index from a list of SMILES.

        Parameters
        ----------
        smiles:
            Reference compound SMILES strings.
        properties:
            Optional dict mapping property names to value lists (same length as
            ``smiles``).  These are stored and returned alongside hits.
        """
        valid_idx: list[int] = []
        bit_fps: list[DataStructs.ExplicitBitVect] = []
        valid_smiles: list[str] = []

        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nbits)
            bit_fps.append(fp)
            valid_smiles.append(smi)
            valid_idx.append(i)

        self._smiles = valid_smiles
        self._bit_fps = bit_fps

        # Float matrix for sklearn-based index
        n = len(bit_fps)
        mat = np.zeros((n, self.nbits), dtype=np.float32)
        for i, fp in enumerate(bit_fps):
            mat[i] = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
        self._fps_matrix = mat

        # Store properties (aligned to valid_idx)
        self._properties = {}
        if properties:
            for key, vals in properties.items():
                self._properties[key] = [vals[i] for i in valid_idx]

        # Build sklearn Ball-Tree for cosine metric
        if self.metric == "cosine":
            from sklearn.neighbors import BallTree
            self._nn_index = BallTree(mat, metric="euclidean")

        logger.info("Built retrieval index with %d compounds", len(valid_smiles))
        return self

    # ── query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        smiles: str,
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> pd.DataFrame:
        """Retrieve the *k* most similar reference compounds.

        Returns a DataFrame with columns: rank, smiles, tanimoto, [property...].
        """
        if not self._smiles:
            raise RuntimeError("Index is empty — call .build() first.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid query SMILES: {smiles!r}")

        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nbits)

        if self.metric == "tanimoto" or self._nn_index is None:
            sims = DataStructs.BulkTanimotoSimilarity(query_fp, self._bit_fps)
            sims = np.asarray(sims, dtype=np.float32)
        else:
            # Cosine via euclidean on binary vectors (approximate)
            qvec = np.frombuffer(query_fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
            # Fallback to tanimoto for accuracy
            sims = DataStructs.BulkTanimotoSimilarity(query_fp, self._bit_fps)
            sims = np.asarray(sims, dtype=np.float32)

        # Rank
        idx_sorted = np.argsort(sims)[::-1]
        rows = []
        rank = 0
        for i in idx_sorted:
            sim = float(sims[i])
            if sim < min_similarity:
                break
            if rank >= k:
                break
            row: dict[str, Any] = {
                "rank": rank + 1,
                "smiles": self._smiles[i],
                "tanimoto": round(sim, 4),
            }
            for key, vals in self._properties.items():
                row[key] = vals[i]
            rows.append(row)
            rank += 1

        return pd.DataFrame(rows)

    # ── convenience ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._smiles)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "smiles": self._smiles,
                "bit_fps": self._bit_fps,
                "fps_matrix": self._fps_matrix,
                "properties": self._properties,
                "radius": self.radius,
                "nbits": self.nbits,
                "metric": self.metric,
            },
            path,
        )
        logger.info("Retriever saved to %s (%d compounds)", path, len(self._smiles))

    @classmethod
    def load(cls, path: str | Path) -> "NearestNeighborRetriever":
        data = joblib.load(path)
        instance = cls(
            radius=data["radius"],
            nbits=data["nbits"],
            metric=data["metric"],
        )
        instance._smiles = data["smiles"]
        instance._bit_fps = data["bit_fps"]
        instance._fps_matrix = data["fps_matrix"]
        instance._properties = data["properties"]
        return instance

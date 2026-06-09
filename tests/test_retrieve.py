"""Tests for mol_pilot.retrieve"""
import tempfile
from pathlib import Path

import pytest
import numpy as np

from mol_pilot.retrieve import NearestNeighborRetriever

LIBRARY_SMILES = [
    "c1ccccc1",
    "CC(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "c1ccc(cc1)Cl",
    "c1ccc(cc1)F",
    "CCO",
    "CCCC",
    "c1ccncc1",
    "CC1=CC=CC=C1",
]
LIBRARY_PROPS = {"logS": [1.6, 0.3, -3.6, -1.3, 0.8, 1.4, 1.3, 0.5, 0.9, 0.8]}


class TestNearestNeighborRetriever:
    @pytest.fixture
    def retriever(self):
        r = NearestNeighborRetriever()
        r.build(LIBRARY_SMILES, LIBRARY_PROPS)
        return r

    def test_build_stores_smiles(self, retriever):
        assert len(retriever) == len(LIBRARY_SMILES)

    def test_query_returns_k_rows(self, retriever):
        df = retriever.query("c1ccccc1", k=3)
        assert len(df) == 3

    def test_query_top_hit_is_self(self, retriever):
        # Querying benzene should return benzene as the closest
        df = retriever.query("c1ccccc1", k=1)
        assert df.iloc[0]["tanimoto"] == pytest.approx(1.0, abs=1e-4)

    def test_query_columns(self, retriever):
        df = retriever.query("c1ccccc1", k=3)
        assert "smiles" in df.columns
        assert "tanimoto" in df.columns
        assert "logS" in df.columns

    def test_similarity_descending(self, retriever):
        df = retriever.query("c1ccccc1", k=5)
        sims = df["tanimoto"].tolist()
        assert sims == sorted(sims, reverse=True)

    def test_save_load_roundtrip(self, retriever):
        df_before = retriever.query("c1ccccc1", k=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retriever.joblib"
            retriever.save(path)
            loaded = NearestNeighborRetriever.load(path)
        df_after = loaded.query("c1ccccc1", k=5)
        assert list(df_before.smiles) == list(df_after.smiles)

    def test_query_min_similarity(self, retriever):
        df = retriever.query("c1ccccc1", k=10, min_similarity=0.5)
        if not df.empty:
            assert (df.tanimoto >= 0.5).all()

    def test_invalid_query_raises(self, retriever):
        with pytest.raises(ValueError):
            retriever.query("invalid_smiles")

    def test_empty_retriever_raises(self):
        r = NearestNeighborRetriever()
        with pytest.raises(RuntimeError):
            r.query("c1ccccc1")

"""Tests for mol_pilot.design"""
import numpy as np
import pytest

from mol_pilot.design import generate_analogs, _apply_transforms
from mol_pilot.models import LightGBMPredictor

TRAIN_SMILES = [
    "c1ccccc1", "CC(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccc(cc1)Cl", "c1ccc(cc1)F",
    "CCO", "CCCC", "c1ccncc1", "CC1=CC=CC=C1",
]
TRAIN_Y = np.array([1.6, 0.3, -3.6, -1.3, 0.8, 1.4, 1.3, 0.5, 0.9, 0.8])


@pytest.fixture
def predictor():
    pred = LightGBMPredictor()
    pred.fit(TRAIN_SMILES, TRAIN_Y)
    return pred


class TestApplyTransforms:
    def test_returns_list(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        results = _apply_transforms(mol)
        assert isinstance(results, list)

    def test_products_are_valid(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        results = _apply_transforms(mol)
        for name, product in results:
            smi = Chem.MolToSmiles(product)
            assert Chem.MolFromSmiles(smi) is not None, f"Invalid product for {name}"


class TestGenerateAnalogs:
    def test_returns_dataframe(self, predictor):
        import pandas as pd
        df = generate_analogs("c1ccccc1", predictor, n_analogs=10, include_mmp=False)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, predictor):
        # Use a richer query so SMARTS transforms produce analogs with Tanimoto > 0.3
        df = generate_analogs("Cc1ccc(N)cc1", predictor, n_analogs=10,
                               include_mmp=False, min_tanimoto=0.1)
        assert not df.empty, "Expected at least one analog for 4-methylaniline"
        expected_cols = {"smiles", "transform", "predicted_property", "delta_property",
                         "tanimoto_to_query", "sa_score"}
        assert expected_cols.issubset(set(df.columns))

    def test_no_duplicate_smiles(self, predictor):
        df = generate_analogs("Cc1ccc(N)cc1", predictor, n_analogs=20,
                               include_mmp=False, min_tanimoto=0.1)
        if not df.empty:
            assert df.smiles.nunique() == len(df)

    def test_tanimoto_in_range(self, predictor):
        df = generate_analogs("Cc1ccc(N)cc1", predictor, n_analogs=10,
                               min_tanimoto=0.1, max_tanimoto=0.99, include_mmp=False)
        if not df.empty:
            assert df.tanimoto_to_query.between(0.1, 0.99).all()

    def test_sa_score_filtered(self, predictor):
        df = generate_analogs("Cc1ccc(N)cc1", predictor, n_analogs=10,
                               max_sa=5.0, include_mmp=False, min_tanimoto=0.1)
        if not df.empty:
            assert (df.sa_score <= 5.0).all()

    def test_invalid_smiles_raises(self, predictor):
        with pytest.raises(ValueError):
            generate_analogs("invalid_smiles", predictor)

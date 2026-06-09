"""Tests for mol_pilot.interpret"""
import numpy as np
import pytest

from mol_pilot.featurize import ECFPFeaturizer
from mol_pilot.interpret import (
    ActivityCliff,
    SHAPInterpreter,
    activity_cliff_score,
    get_bit_atom_map,
    map_shap_to_atoms,
    screen_for_cliffs,
    tanimoto_similarity,
)
from mol_pilot.models import LightGBMPredictor

SMILES = [
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
Y = np.array([1.6, 0.3, -3.6, -1.3, 0.8, 1.4, 1.3, 0.5, 0.9, 0.8])


class TestBitAtomMap:
    def test_returns_dict(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        bi = get_bit_atom_map(mol)
        assert isinstance(bi, dict)
        assert all(isinstance(k, int) for k in bi)
        assert all(isinstance(v, list) for v in bi.values())

    def test_atoms_in_range(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CC(=O)O")
        bi = get_bit_atom_map(mol)
        n_atoms = mol.GetNumAtoms()
        for atoms in bi.values():
            assert all(0 <= a < n_atoms for a in atoms)


class TestMapShapToAtoms:
    def test_output_shape(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        shap_vals = np.random.randn(2048)
        weights = map_shap_to_atoms(mol, shap_vals)
        assert weights.shape == (mol.GetNumAtoms(),)

    def test_zero_shap_gives_zero_weights(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        shap_vals = np.zeros(2048)
        weights = map_shap_to_atoms(mol, shap_vals)
        assert np.all(weights == 0)


class TestTanimotoSimilarity:
    def test_self_similarity_is_one(self):
        sim = tanimoto_similarity("c1ccccc1", "c1ccccc1")
        assert abs(sim - 1.0) < 1e-6

    def test_different_molecules(self):
        sim = tanimoto_similarity("c1ccccc1", "CC(=O)O")
        assert 0.0 <= sim < 1.0

    def test_invalid_smiles_returns_zero(self):
        assert tanimoto_similarity("invalid", "c1ccccc1") == 0.0


class TestActivityCliff:
    def test_cliff_detected(self):
        # Fluorobenzene vs chlorobenzene: structurally very similar, large property delta
        result = activity_cliff_score(
            "c1ccc(F)cc1", "c1ccc(Cl)cc1",
            0.0, 5.0,
            sim_threshold=0.25,   # actual Tanimoto ≈ 0.27
            delta_threshold=1.0,
        )
        assert isinstance(result, ActivityCliff)
        assert result.is_cliff is True

    def test_no_cliff_low_delta(self):
        result = activity_cliff_score(
            "c1ccccc1", "c1ccc(cc1)Cl",
            0.0, 0.2,  # small delta
            sim_threshold=0.30,
            delta_threshold=1.0,
        )
        assert result.is_cliff is False

    def test_no_cliff_low_similarity(self):
        result = activity_cliff_score(
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen — dissimilar
            0.0, 5.0,
            sim_threshold=0.90,  # high threshold
            delta_threshold=1.0,
        )
        assert result.is_cliff is False

    def test_screen_finds_cliffs(self):
        smi = ["c1ccccc1", "c1ccc(cc1)F", "CCCCCC"]
        props = [0.0, 4.0, -1.0]
        cliffs = screen_for_cliffs(smi, props, sim_threshold=0.3, delta_threshold=1.0)
        assert isinstance(cliffs, list)


class TestSHAPInterpreter:
    @pytest.fixture
    def trained_predictor(self):
        pred = LightGBMPredictor()
        pred.fit(SMILES, Y)
        return pred

    def test_explain_shape(self, trained_predictor):
        interp = SHAPInterpreter(trained_predictor, background_smiles=SMILES[:5])
        sv = interp.explain("c1ccccc1")
        assert sv.shape == (2048,)

    def test_atom_weights_shape(self, trained_predictor):
        interp = SHAPInterpreter(trained_predictor, background_smiles=SMILES[:5])
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        weights = interp.atom_weights("c1ccccc1")
        assert weights.shape == (mol.GetNumAtoms(),)

    def test_top_features_count(self, trained_predictor):
        interp = SHAPInterpreter(trained_predictor, background_smiles=SMILES[:5])
        feats = interp.top_features("c1ccccc1", n=5)
        assert len(feats) == 5
        assert all("bit" in f and "shap_value" in f for f in feats)

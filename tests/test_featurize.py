"""Tests for mol_pilot.featurize"""
import numpy as np
import pytest

from mol_pilot.featurize import (
    CombinedFeaturizer,
    ECFPFeaturizer,
    RDKitDescriptorFeaturizer,
)

SMILES = [
    "c1ccccc1",
    "CC(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
]


class TestECFPFeaturizer:
    def test_output_shape(self):
        featurizer = ECFPFeaturizer(radius=2, nbits=2048)
        X = featurizer.transform(SMILES)
        assert X.shape == (len(SMILES), 2048)

    def test_binary_values(self):
        X = ECFPFeaturizer().transform(SMILES)
        assert set(np.unique(X)).issubset({0.0, 1.0})

    def test_custom_nbits(self):
        X = ECFPFeaturizer(nbits=512).transform(SMILES)
        assert X.shape == (len(SMILES), 512)

    def test_invalid_smiles_row_is_zeros(self):
        X = ECFPFeaturizer().transform(["invalid_smiles"])
        assert np.all(X[0] == 0)

    def test_feature_names_length(self):
        feat = ECFPFeaturizer(nbits=1024)
        assert len(feat.get_feature_names()) == 1024

    def test_feature_names_prefix(self):
        names = ECFPFeaturizer().get_feature_names()
        assert names[0].startswith("ecfp_")

    def test_bit_info_keys_are_ints(self):
        feat = ECFPFeaturizer()
        bi = feat.get_bit_info("c1ccccc1")
        for k in bi:
            assert isinstance(k, int)

    def test_different_radius(self):
        X2 = ECFPFeaturizer(radius=2).transform(SMILES)
        X3 = ECFPFeaturizer(radius=3).transform(SMILES)
        # Different radii should produce different fingerprints
        assert not np.array_equal(X2, X3)


class TestRDKitDescriptorFeaturizer:
    def test_output_shape(self):
        feat = RDKitDescriptorFeaturizer()
        X = feat.transform(SMILES[:2])
        n_desc = len(feat.get_feature_names())
        assert X.shape == (2, n_desc)

    def test_finite_values(self):
        feat = RDKitDescriptorFeaturizer()
        X = feat.transform(SMILES)
        assert np.all(np.isfinite(X))

    def test_custom_descriptors(self):
        feat = RDKitDescriptorFeaturizer(descriptor_names=["MolWt", "MolLogP"])
        X = feat.transform(["c1ccccc1"])
        assert X.shape == (1, 2)
        # benzene MW ≈ 78
        assert 77 < X[0, 0] < 79


class TestCombinedFeaturizer:
    def test_shape(self):
        feat = CombinedFeaturizer(
            ECFPFeaturizer(nbits=512),
            RDKitDescriptorFeaturizer(descriptor_names=["MolWt", "MolLogP"]),
        )
        X = feat.transform(SMILES)
        assert X.shape == (len(SMILES), 514)

    def test_feature_names_concatenated(self):
        feat = CombinedFeaturizer(
            ECFPFeaturizer(nbits=256),
            RDKitDescriptorFeaturizer(descriptor_names=["MolWt"]),
        )
        names = feat.get_feature_names()
        assert len(names) == 257
        assert names[0].startswith("ecfp_")
        assert names[-1] == "MolWt"

    def test_requires_two_featurizers(self):
        with pytest.raises(ValueError):
            CombinedFeaturizer(ECFPFeaturizer())

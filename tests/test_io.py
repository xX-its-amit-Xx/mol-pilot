"""Tests for mol_pilot.io"""
import pytest
from rdkit import Chem

from mol_pilot.io import (
    mol_to_smiles,
    parse_smiles,
    parse_sdf,
    standardize_mol,
    validate_smiles,
)

VALID_SMILES = [
    "c1ccccc1",                   # benzene
    "CC(=O)O",                    # acetic acid
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
]

INVALID_SMILES = ["not_a_smiles", "", "   ", "C(C(C(C"]


class TestParseSMILES:
    def test_valid_returns_mol(self):
        for smi in VALID_SMILES:
            mol = parse_smiles(smi)
            assert mol is not None, f"Expected valid mol for {smi}"

    def test_invalid_returns_none(self):
        for smi in INVALID_SMILES:
            assert parse_smiles(smi) is None, f"Expected None for {smi!r}"

    def test_standardize_flag(self):
        # With a salt: only the organic part should survive
        mol_std = parse_smiles("c1ccccc1.O", standardize=True)
        mol_raw = parse_smiles("c1ccccc1.O", standardize=False)
        assert mol_std is not None
        # Standardised mol should not contain disconnected fragments
        smi_std = Chem.MolToSmiles(mol_std)
        assert "." not in smi_std


class TestValidateSMILES:
    def test_valid(self):
        for smi in VALID_SMILES:
            assert validate_smiles(smi) is True

    def test_invalid(self):
        for smi in INVALID_SMILES:
            assert validate_smiles(smi) is False


class TestStandardizeMol:
    def test_removes_salt(self):
        mol = Chem.MolFromSmiles("c1ccccc1.[Na+].[Cl-]")
        std = standardize_mol(mol)
        assert std is not None
        smi = Chem.MolToSmiles(std)
        assert "." not in smi

    def test_none_input(self):
        assert standardize_mol(None) is None


class TestMolToSMILES:
    def test_canonical(self):
        # Two representations of the same molecule → same canonical SMILES
        smi1 = mol_to_smiles(Chem.MolFromSmiles("OCC"))
        smi2 = mol_to_smiles(Chem.MolFromSmiles("CCO"))
        assert smi1 == smi2

    def test_none_returns_none(self):
        assert mol_to_smiles(None) is None

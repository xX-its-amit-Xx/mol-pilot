"""
SMILES/SDF parsing, validation, and standardisation.

All public functions accept raw strings and return RDKit Mol objects or
canonical SMILES. Failures are logged and return None rather than raising,
so batch pipelines can process partial datasets gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

# ── standardisation pipeline (built once, reused) ────────────────────────────

_LARGEST_FRAG = rdMolStandardize.LargestFragmentChooser()
_NORMALIZER = rdMolStandardize.Normalizer()
_UNCHARGER = rdMolStandardize.Uncharger()
_TE = rdMolStandardize.TautomerEnumerator()


def parse_smiles(smiles: str, standardize: bool = True) -> Chem.Mol | None:
    """Parse a SMILES string and optionally standardise it.

    Returns None if the SMILES is invalid.
    """
    if not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.debug("Invalid SMILES: %s", smiles)
        return None
    if standardize:
        mol = standardize_mol(mol)
    return mol


def validate_smiles(smiles: str) -> bool:
    """Return True if SMILES is parseable by RDKit."""
    return parse_smiles(smiles, standardize=False) is not None


def standardize_mol(mol: Chem.Mol) -> Chem.Mol | None:
    """Remove salts, normalise, uncharge, and canonicalise a molecule."""
    if mol is None:
        return None
    try:
        mol = _LARGEST_FRAG.choose(mol)
        mol = _NORMALIZER.normalize(mol)
        mol = _UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception as exc:
        logger.debug("Standardisation failed: %s", exc)
        return None


def mol_to_smiles(mol: Chem.Mol | None) -> str | None:
    """Convert a Mol to a canonical SMILES string."""
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def parse_sdf(path: str | Path, standardize: bool = True) -> list[Chem.Mol]:
    """Parse an SDF file and return a list of valid Mol objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
    mols: list[Chem.Mol] = []
    for mol in supplier:
        if mol is None:
            continue
        if standardize:
            mol = standardize_mol(mol)
        if mol is not None:
            mols.append(mol)
    logger.info("Loaded %d valid molecules from %s", len(mols), path)
    return mols


def iter_smiles_file(
    path: str | Path, smiles_col: int = 0, skip_header: bool = False
) -> Generator[tuple[str, Chem.Mol | None], None, None]:
    """Yield (smiles, mol) pairs from a plain-text SMILES file.

    Lines starting with '#' are skipped. Set *smiles_col* if the SMILES is not
    the first whitespace-delimited field.
    """
    path = Path(path)
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if skip_header and i == 0:
                continue
            parts = line.split()
            if smiles_col >= len(parts):
                continue
            smi = parts[smiles_col]
            yield smi, parse_smiles(smi)


def smiles_to_inchi(smiles: str) -> str | None:
    """Convert SMILES to InChI (requires RDKit InChI support)."""
    mol = parse_smiles(smiles, standardize=False)
    if mol is None:
        return None
    try:
        from rdkit.Chem.inchi import MolToInchi
        return MolToInchi(mol)
    except ImportError:
        logger.warning("RDKit InChI support not available")
        return None

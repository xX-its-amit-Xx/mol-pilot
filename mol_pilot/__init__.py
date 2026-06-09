"""
MolPilot — a medicinal-chemist-facing molecular design copilot.

Core pipeline:
  io          → parse/standardise SMILES/SDF
  featurize   → ECFP fingerprints + RDKit descriptors
  models      → model-agnostic Predictor protocol (LightGBM/sklearn adapters)
  interpret   → SHAP attributions mapped to atoms, activity-cliff detection
  design      → MMP + fragment-recombination analog generation
  retrieve    → nearest-neighbour similar-compound retrieval
  app         → Streamlit UI
  cli         → command-line batch scoring & design
"""

from mol_pilot.io import parse_smiles, parse_sdf, standardize_mol, mol_to_smiles, validate_smiles
from mol_pilot.featurize import ECFPFeaturizer, RDKitDescriptorFeaturizer, CombinedFeaturizer
from mol_pilot.models import LightGBMPredictor, SklearnPredictor, load_predictor, save_predictor
from mol_pilot.interpret import SHAPInterpreter, activity_cliff_score
from mol_pilot.design import generate_analogs
from mol_pilot.retrieve import NearestNeighborRetriever

__version__ = "0.1.0"
__all__ = [
    "parse_smiles",
    "parse_sdf",
    "standardize_mol",
    "mol_to_smiles",
    "validate_smiles",
    "ECFPFeaturizer",
    "RDKitDescriptorFeaturizer",
    "CombinedFeaturizer",
    "LightGBMPredictor",
    "SklearnPredictor",
    "load_predictor",
    "save_predictor",
    "SHAPInterpreter",
    "activity_cliff_score",
    "generate_analogs",
    "NearestNeighborRetriever",
]

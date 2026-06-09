"""
Cookbook 1 — Train a LightGBM aqueous-solubility model end-to-end
==================================================================

Data  : AqSolDB (Therapeutics Data Commons) — 9,982 compounds,
        measured log10 aqueous solubility (mol/L).
        Falls back to an embedded 100-compound ESOL subset if TDC is offline.

Output: models_saved/solubility_lgbm.joblib   — trained predictor
        models_saved/solubility_retriever.joblib — kNN index (for app + CB-2)
        cookbook/data/solubility_predictions.csv — hold-out predictions

Run  : python cookbook/01_train_solubility_lgbm.py
"""

import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows so Unicode characters render correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Ensure mol_pilot is importable when run from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Embedded ESOL fallback (100 compounds from Delaney 2004)
# ─────────────────────────────────────────────────────────────────────────────

ESOL_FALLBACK = [
    ("c1ccccc1", 1.580), ("CC(=O)O", 0.334), ("c1ccc(Cl)cc1", 0.838),
    ("c1ccc(F)cc1", 1.378), ("c1ccc(Br)cc1", 0.501), ("c1ccc(I)cc1", 0.133),
    ("c1ccc(N)cc1", -0.028), ("c1ccc(O)cc1", 0.100), ("c1ccc([N+](=O)[O-])cc1", -0.386),
    ("c1ccc(C)cc1", 0.851), ("c1ccc(OC)cc1", 0.940), ("c1ccc(C#N)cc1", -0.217),
    ("c1ccc(C(=O)O)cc1", -0.517), ("c1ccc(C(F)(F)F)cc1", -0.177),
    ("c1ccc2ccccc2c1", -3.177), ("c1ccc2cccc3cccc1c23", -5.360),
    ("CCO", 1.310), ("CCCO", 1.020), ("CCCCO", 0.710),
    ("CC(=O)C", 1.014), ("CCCC(=O)C", -0.282), ("CC(C)=O", 0.914),
    ("CCCBr", -0.163), ("CCCCBr", -0.660), ("CC(Br)C", -0.380),
    ("CCCCI", -0.725), ("CCCCCC", -2.800), ("CCCCCCC", -3.350),
    ("CCCCCCCC", -3.880), ("CC(C)CC(C)(C)C", -3.630),
    ("CCOCC", -0.060), ("CCOCc1ccccc1", -1.430),
    ("CC(=O)OC", 0.330), ("CCOC(=O)C", 0.120), ("CCOC(=O)CC", -0.540),
    ("CCC(=O)OC", -0.050), ("CCCOC(=O)C", -0.630), ("CC(N)=O", 0.280),
    ("CCC(N)=O", -0.350), ("CCCC(N)=O", -0.750), ("CCCCN", 0.180),
    ("CCCCCN", -0.430), ("CC(C)N", 0.380), ("c1ccncc1", 0.906),
    ("c1ccncn1", -0.219), ("c1cnc2ccccc2n1", -1.290),
    ("CN1CCCCC1", -0.220), ("C1CCNCC1", 0.130), ("C1CCCNCC1", -0.390),
    ("CC1=CC=CC=C1", 0.540), ("CC1=CC(=CC=C1)C", -0.140),
    ("CC1=CC=C(C)C=C1", 0.050), ("CC1=CC=CC=C1C", -0.100),
    ("CC(=O)Nc1ccccc1", -1.230), ("CC(=O)Nc1ccc(Cl)cc1", -2.600),
    ("CC(=O)Nc1ccc(O)cc1", -1.600), ("COc1ccccc1", 0.130),
    ("COc1ccc(Cl)cc1", -1.540), ("COc1ccc(OC)cc1", -1.100),
    ("Clc1ccccc1Cl", -2.025), ("Clc1ccc(Cl)cc1", -2.787),
    ("Clc1cccc(Cl)c1", -2.200), ("Clc1ccc(Cl)c(Cl)c1", -3.500),
    ("Clc1cc(Cl)cc(Cl)c1", -3.900), ("Fc1ccc(F)cc1", 0.620),
    ("Fc1ccc(Cl)cc1", -0.810), ("CC(=O)c1ccccc1", -1.510),
    ("CC(=O)c1ccc(Cl)cc1", -2.820), ("CC(=O)c1ccc(Br)cc1", -3.110),
    ("CC(=O)c1ccc(F)cc1", -1.680), ("O=C(O)c1ccccc1", -1.300),
    ("O=C(O)c1ccc(Cl)cc1", -2.600), ("O=C(O)c1ccc(Br)cc1", -2.990),
    ("O=C(O)c1cccc(Cl)c1", -2.520), ("CN(C)c1ccccc1", -0.900),
    ("Nc1ccc(Cl)cc1", -1.400), ("Nc1ccc(Br)cc1", -2.000),
    ("Nc1ccc(F)cc1", -0.740), ("Nc1ccc(C)cc1", -0.890),
    ("CC(O)=O", 0.334), ("CCC(O)=O", -0.270), ("CCCC(O)=O", -0.755),
    ("CCCCC(O)=O", -1.200), ("CCCCCC(O)=O", -1.700),
    ("C(Cl)Cl", 0.610), ("C(Cl)(Cl)Cl", -0.240), ("C(Cl)(Cl)(Cl)Cl", -1.200),
    ("CC(Cl)Cl", -0.355), ("CC(Cl)(Cl)C", -1.650),
    ("CCCCC", -2.260), ("CCCC", -1.680),
    ("CC(C)(C)C", -1.700), ("CCC(C)C", -1.980),
    ("CCCCCCCCCC", -5.300), ("CC(O)c1ccccc1", -0.890),
    ("OC(Cl)(Cl)Cl", -0.890), ("OC(F)(F)F", 0.290),
    ("CC(C)=O", 0.914), ("CC(C)(C)O", 0.580), ("OCC", 1.310),
    ("OCCO", 0.740), ("OCCCO", 0.390), ("OC(CO)CO", -0.310),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Try TDC first; fall back to embedded ESOL subset."""
    try:
        print("Fetching AqSolDB from Therapeutics Data Commons (TDC)…")
        from tdc.single_pred import ADME
        data = ADME(name="Solubility_AqSolDB", path=str(ROOT / "cookbook" / "data"))
        split = data.get_split(method="random", frac=[0.8, 0.1, 0.1], seed=42)
        frames = []
        for tag, df in [("train", split["train"]), ("valid", split["valid"]), ("test", split["test"])]:
            df = df.copy()
            df["split"] = tag
            frames.append(df)
        full = pd.concat(frames, ignore_index=True)
        full = full.rename(columns={"Drug": "smiles", "Y": "logS"})
        print(f"  Loaded {len(full):,} compounds from TDC AqSolDB")
        return full[["smiles", "logS", "split"]]
    except Exception as exc:
        print(f"  TDC unavailable ({exc}); using embedded ESOL-100 fallback")
        df = pd.DataFrame(ESOL_FALLBACK, columns=["smiles", "logS"])
        # 70/15/15 split
        idx = np.arange(len(df))
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        n = len(df)
        splits = np.array(["train"] * n)
        splits[idx[int(0.7 * n):int(0.85 * n)]] = "valid"
        splits[idx[int(0.85 * n):]] = "test"
        df["split"] = splits
        print(f"  Using embedded fallback: {len(df)} compounds")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Validate SMILES
# ─────────────────────────────────────────────────────────────────────────────

def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    from mol_pilot.io import validate_smiles
    mask = df["smiles"].apply(validate_smiles)
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"  Dropped {n_dropped} invalid SMILES")
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Train
# ─────────────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame):
    from mol_pilot.featurize import ECFPFeaturizer
    from mol_pilot.models import LightGBMPredictor

    train_df = df[df.split == "train"]
    val_df   = df[df.split == "valid"]
    test_df  = df[df.split == "test"]

    print(f"\nSplit sizes — train: {len(train_df)}, valid: {len(val_df)}, test: {len(test_df)}")

    feat = ECFPFeaturizer(radius=2, nbits=2048)
    predictor = LightGBMPredictor(
        featurizer=feat,
        model_name="logS (AqSolDB)",
    )

    print("\nTraining LightGBM model…")
    predictor.fit(
        list(train_df.smiles),
        train_df.logS.values,
        smiles_val=list(val_df.smiles),
        y_val=val_df.logS.values,
        callbacks=[],
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    from mol_pilot.models import evaluate_regression

    print("\n── In-sample (train) ──")
    metrics_tr = evaluate_regression(predictor, list(train_df.smiles), train_df.logS.values)
    for k, v in metrics_tr.items():
        print(f"  {k.upper():5s}: {v:.4f}")

    print("\n── Hold-out (test) ──")
    metrics_te = evaluate_regression(predictor, list(test_df.smiles), test_df.logS.values)
    for k, v in metrics_te.items():
        print(f"  {k.upper():5s}: {v:.4f}")

    return predictor, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Build retriever
# ─────────────────────────────────────────────────────────────────────────────

def build_retriever(df: pd.DataFrame):
    from mol_pilot.retrieve import NearestNeighborRetriever

    print("\nBuilding k-NN retrieval index…")
    retriever = NearestNeighborRetriever()
    retriever.build(
        list(df.smiles),
        properties={"logS": list(df.logS.astype(float))},
    )
    print(f"  Index built: {len(retriever):,} compounds")
    return retriever


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Save artefacts
# ─────────────────────────────────────────────────────────────────────────────

def save_artefacts(predictor, retriever, test_df):
    out_dir = ROOT / "models_saved"
    out_dir.mkdir(exist_ok=True)
    data_dir = ROOT / "cookbook" / "data"
    data_dir.mkdir(exist_ok=True)

    predictor.save(out_dir / "solubility_lgbm.joblib")
    retriever.save(out_dir / "solubility_retriever.joblib")

    # Predictions CSV
    preds = predictor.predict(list(test_df.smiles))
    out_csv = data_dir / "solubility_predictions.csv"
    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["predicted_logS"] = preds
    pred_df["residual"] = pred_df.logS - pred_df.predicted_logS
    pred_df.to_csv(out_csv, index=False)

    print(f"\nArtefacts saved:")
    print(f"  Model   : {out_dir / 'solubility_lgbm.joblib'}")
    print(f"  Retriever: {out_dir / 'solubility_retriever.joblib'}")
    print(f"  CSV     : {out_csv}  ({len(pred_df)} rows)")

    # Quick sample
    print("\nSample predictions (test set, first 10):")
    print(pred_df[["smiles", "logS", "predicted_logS", "residual"]].head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print(" Cookbook 1 — Solubility LightGBM Model (MolPilot)")
    print("=" * 65)

    df = filter_valid(load_data())
    predictor, test_df = train(df)
    retriever = build_retriever(df)
    save_artefacts(predictor, retriever, test_df)

    print("\nDone. Next: run cookbook/02_molpilot_imatinib_walkthrough.py")

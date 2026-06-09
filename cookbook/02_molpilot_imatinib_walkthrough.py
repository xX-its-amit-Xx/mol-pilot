"""
Cookbook 2 — MolPilot walkthrough on imatinib (Gleevec, STI-571)
=================================================================

Imatinib is the first-in-class BCR-ABL kinase inhibitor that transformed
CML treatment. Here we use it as a worked example for the full MolPilot
pipeline:

  Step 1 — Predict aqueous solubility (logS) using the CB-1 LightGBM model.
  Step 2 — Explain prediction with SHAP atom-level attribution.
  Step 3 — Retrieve the 10 most similar compounds from the AqSolDB index.
  Step 4 — Generate ranked analog suggestions (SMARTS transforms + MMP swaps).
  Step 5 — Export everything to CSV (Benchling / DataWarrior ready).
  Step 6 — Demonstrate CLI batch-scoring on a small SMILES file.

Prerequisite: run cookbook/01_train_solubility_lgbm.py first.

Run  : python cookbook/02_molpilot_imatinib_walkthrough.py
"""

import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Imatinib SMILES (canonical, from PubChem CID 5291)
# ─────────────────────────────────────────────────────────────────────────────
IMATINIB_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
IMATINIB_NAME   = "Imatinib (Gleevec)"

MODEL_PATH     = ROOT / "models_saved" / "solubility_lgbm.joblib"
RETRIEVER_PATH = ROOT / "models_saved" / "solubility_retriever.joblib"
OUT_DIR        = ROOT / "cookbook" / "data"

sep = "=" * 65


def load_artefacts():
    from mol_pilot.models import load_predictor
    from mol_pilot.retrieve import NearestNeighborRetriever

    if not MODEL_PATH.exists():
        sys.exit("Model not found — run cookbook/01_train_solubility_lgbm.py first.")

    predictor = load_predictor(MODEL_PATH)
    retriever = NearestNeighborRetriever.load(RETRIEVER_PATH) if RETRIEVER_PATH.exists() else None
    print(f"Model loaded : {predictor.model_name}")
    if retriever:
        print(f"Retriever    : {len(retriever):,} compounds")
    return predictor, retriever


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Prediction
# ─────────────────────────────────────────────────────────────────────────────

def step1_predict(predictor, smiles: str) -> float:
    print(f"\n{sep}")
    print(" Step 1 — Property Prediction")
    print(sep)
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED as rdQED

    mol = Chem.MolFromSmiles(smiles)
    pred = float(predictor.predict([smiles])[0])

    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    qed  = float(rdQED.qed(mol))

    print(f"\nMolecule      : {IMATINIB_NAME}")
    print(f"SMILES        : {smiles}")
    print(f"\nPredicted logS: {pred:+.3f} (mol/L)  [log10 aqueous solubility]")
    print(f"  → Solubility class: {'Low (<-4)' if pred < -4 else 'Moderate (-4 to -2)' if pred < -2 else 'High (>-2)'}")
    print(f"\nPhysico-chemical profile:")
    print(f"  MW     : {mw:.1f} Da")
    print(f"  LogP   : {logp:.2f}")
    print(f"  HBD    : {hbd}")
    print(f"  HBA    : {hba}")
    print(f"  TPSA   : {tpsa:.1f} Å²")
    print(f"  QED    : {qed:.3f}")
    print(f"\n  Lipinski Ro5: {'PASS' if mw<=500 and logp<=5 and hbd<=5 and hba<=10 else 'FAIL'}")
    print(f"  Veber   (TPSA≤140, RotB≤10): {'PASS' if tpsa<=140 else 'FAIL'}")
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — SHAP interpretation
# ─────────────────────────────────────────────────────────────────────────────

def step2_shap(predictor, smiles: str):
    print(f"\n{sep}")
    print(" Step 2 — SHAP Atom-Level Attribution")
    print(sep)

    from mol_pilot.interpret import SHAPInterpreter

    interpreter = SHAPInterpreter(predictor)
    shap_vals = interpreter.explain(smiles)
    atom_weights = interpreter.atom_weights(smiles)
    top = interpreter.top_features(smiles, n=10)

    print(f"\nTop-10 fingerprint bits by |SHAP| value:")
    print(f"  {'Bit':>6}  {'SHAP':>9}  Direction")
    for f in top:
        print(f"  {f['bit']:>6}  {f['shap_value']:>+9.4f}  {f['direction']}")

    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    print(f"\nPer-atom weights (top 5 by |weight|):")
    pairs = sorted(enumerate(atom_weights), key=lambda x: abs(x[1]), reverse=True)[:5]
    for atom_idx, w in pairs:
        sym = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        print(f"  Atom {atom_idx:2d} ({sym}): {w:+.4f}")

    # Save highlighted image
    try:
        img = interpreter.draw_highlighted_mol(smiles, size=(700, 450))
        out_path = OUT_DIR / "imatinib_shap.png"
        img.save(out_path)
        print(f"\nSHAP-highlighted structure saved → {out_path}")
    except Exception as exc:
        print(f"\nImage save skipped: {exc}")

    return shap_vals, atom_weights


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Similar-compound retrieval
# ─────────────────────────────────────────────────────────────────────────────

def step3_retrieve(retriever, smiles: str) -> pd.DataFrame | None:
    print(f"\n{sep}")
    print(" Step 3 — Nearest-Neighbour Similar Compounds")
    print(sep)

    if retriever is None:
        print("Retriever not available — skipping.")
        return None

    df = retriever.query(smiles, k=10, min_similarity=0.10)
    print(f"\n10 most similar compounds in AqSolDB (by ECFP4 Tanimoto):")
    print(df[["rank", "smiles", "tanimoto", "logS"]].to_string(index=False))

    out = OUT_DIR / "imatinib_similar_compounds.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Design suggestions
# ─────────────────────────────────────────────────────────────────────────────

def step4_design(predictor, smiles: str, query_pred: float) -> pd.DataFrame:
    print(f"\n{sep}")
    print(" Step 4 — Ranked Analog Design Suggestions")
    print(sep)

    from mol_pilot.design import generate_analogs

    print("Generating analogs (SMARTS transforms + MMP fragment swaps)…")
    df = generate_analogs(
        smiles,
        predictor,
        n_analogs=30,
        include_mmp=True,
        optimize_direction="maximize",   # higher logS = more soluble = better
        min_tanimoto=0.25,
        max_sa=8.0,
    )

    if df.empty:
        print("No analogs found above filters.")
        return df

    print(f"\nTop 15 analog suggestions ranked by Δ logS:")
    cols = ["rank", "smiles", "transform", "predicted_property", "delta_property",
            "tanimoto_to_query", "sa_score", "mol_weight", "passes_lipinski"]
    print(df[cols].head(15).to_string(index=False))

    out = OUT_DIR / "imatinib_design_suggestions.csv"
    df.to_csv(out, index=False)
    print(f"\nFull table saved → {out}  (open in DataWarrior or Benchling)")

    # Summary statistics
    print(f"\nDesign space summary:")
    print(f"  Total suggestions    : {len(df)}")
    print(f"  Improving (ΔlogS>0)  : {(df.delta_property > 0).sum()}")
    print(f"  Lipinski-compliant   : {df.passes_lipinski.sum()}")
    print(f"  Median SA score      : {df.sa_score.median():.2f}")
    print(f"  Best predicted logS  : {df.predicted_property.max():.3f}")
    print(f"  Worst predicted logS : {df.predicted_property.min():.3f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — CLI batch scoring demo
# ─────────────────────────────────────────────────────────────────────────────

def step5_cli_demo():
    print(f"\n{sep}")
    print(" Step 5 — CLI Batch Scoring Demo")
    print(sep)
    import subprocess

    # Write a small SMILES file
    demo_smiles = [
        "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1  imatinib",
        "c1ccc2ncccc2c1  quinoline",
        "CC(=O)Oc1ccccc1C(=O)O  aspirin",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C  caffeine",
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C  testosterone",
    ]
    smi_file = OUT_DIR / "demo_batch.smi"
    with open(smi_file, "w") as fh:
        for line in demo_smiles:
            fh.write(line + "\n")

    out_csv = OUT_DIR / "demo_batch_predictions.csv"
    cmd = [
        sys.executable, "-m", "mol_pilot.cli",
        "score", str(smi_file),
        "--model", str(MODEL_PATH),
        "--output", str(out_csv),
        "--descending",
    ]
    print(f"\nRunning: python -m mol_pilot.cli score <file> --model <model>")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[:500])

    if out_csv.exists():
        df = pd.read_csv(out_csv)
        print("\nBatch predictions:")
        print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(sep)
    print(" Cookbook 2 — MolPilot walkthrough on Imatinib (Gleevec)")
    print(sep)

    OUT_DIR.mkdir(exist_ok=True)

    predictor, retriever = load_artefacts()

    query_pred = step1_predict(predictor, IMATINIB_SMILES)
    shap_vals, atom_weights = step2_shap(predictor, IMATINIB_SMILES)
    similar_df = step3_retrieve(retriever, IMATINIB_SMILES)
    design_df  = step4_design(predictor, IMATINIB_SMILES, query_pred)
    step5_cli_demo()

    # ── Master export ─────────────────────────────────────────────────────────
    master = {
        "query_smiles": IMATINIB_SMILES,
        "query_name": IMATINIB_NAME,
        "predicted_logS": round(query_pred, 3),
    }
    print(f"\n{sep}")
    print(" Summary")
    print(sep)
    for k, v in master.items():
        print(f"  {k}: {v}")
    print(f"\nAll outputs in: {OUT_DIR}")
    print("Done. Next: run cookbook/03_activity_cliff_demo.py")

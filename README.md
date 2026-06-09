# MolPilot — Molecular Design Copilot

> **Research tooling only. Not validated for clinical, regulatory, or commercial decision-making.**
> MolPilot is an open-source prototype intended for exploratory medicinal-chemistry research.
> Predictions are probabilistic estimates from data-driven models and carry no certification.

---

## The Problem: Model-to-Decision Gap

Computational teams routinely build accurate property models — solubility, CYP inhibition,
hERG risk, pEC50 — yet bench chemists struggle to act on them. The gap is not predictive quality;
it is **interpretability and workflow integration**.

MolPilot bridges this gap by wrapping any property model behind four chemist-facing lenses:

| Lens | Question answered |
|---|---|
| **Prediction card** | What will this molecule's property be? How does it rank? |
| **SHAP highlights** | *Why* does the model predict that? Which atoms drive the score? |
| **Similar compounds** | What related structures are already in our data? What did they measure? |
| **Design suggestions** | What should I make next? Ranked by predicted gain + synthesisability. |

---

## Installation

```bash
git clone https://github.com/molpilot/molpilot.git
cd molpilot
pip install -e .
# with GNN support:
pip install -e ".[gnn]"
# with TDC data access:
pip install -e ".[tdc]"
```

**Requirements:** Python ≥ 3.11, RDKit ≥ 2023.3, LightGBM ≥ 4.0, scikit-learn ≥ 1.3,
SHAP ≥ 0.43, Streamlit ≥ 1.28, pandas, numpy, plotly.

---

## Quickstart

### 1. Interactive UI

```bash
# Launch the Streamlit app (after training a model — see cookbook 1):
molpilot ui
# or:
streamlit run mol_pilot/app.py
```

Load a trained model from the sidebar, paste a SMILES, and click **Analyse Molecule**.

### 2. CLI — batch score a SMILES file

```bash
molpilot score compounds.smi \
    --model models_saved/solubility_lgbm.joblib \
    --output predictions.csv \
    --descending
```

### 3. CLI — design suggestions for one molecule

```bash
molpilot design "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1" \
    --model models_saved/solubility_lgbm.joblib \
    --n-analogs 30 \
    --output imatinib_suggestions.csv
```

### 4. Python API

```python
from mol_pilot import (
    ECFPFeaturizer, LightGBMPredictor, SHAPInterpreter,
    NearestNeighborRetriever, generate_analogs
)

# Train
feat = ECFPFeaturizer(radius=2, nbits=2048)
model = LightGBMPredictor(featurizer=feat, model_name="logS")
model.fit(train_smiles, train_logS)
model.save("solubility.joblib")

# Predict
preds = model.predict(["CC(=O)Oc1ccccc1C(=O)O"])   # aspirin

# Explain
interp = SHAPInterpreter(model, background_smiles=train_smiles[:100])
weights = interp.atom_weights("CC(=O)Oc1ccccc1C(=O)O")
img = interp.draw_highlighted_mol("CC(=O)Oc1ccccc1C(=O)O")   # PIL Image
img.save("aspirin_shap.png")

# Retrieve
retriever = NearestNeighborRetriever().build(ref_smiles, {"logS": ref_values})
df = retriever.query("CC(=O)Oc1ccccc1C(=O)O", k=10)

# Design
suggestions = generate_analogs("CC(=O)Oc1ccccc1C(=O)O", model, n_analogs=20)
suggestions.to_csv("aspirin_suggestions.csv", index=False)
```

---

## The Predictor Plugin API

Any object satisfying the `Predictor` structural protocol can be used wherever
MolPilot expects a property model:

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Predictor(Protocol):
    featurizer: Featurizer   # used by SHAP interpreter
    model_name: str
    def predict(self, smiles: list[str]) -> np.ndarray: ...
```

**Example — plug in a PyTorch MPNN:**

```python
class MyMPNNPredictor:
    featurizer = ECFPFeaturizer()          # still used by SHAP
    model_name = "MPNN-pKi"

    def __init__(self, checkpoint_path):
        self.net = load_mpnn(checkpoint_path)

    def predict(self, smiles):
        graphs = [mol_to_graph(s) for s in smiles]
        with torch.no_grad():
            return self.net(graphs).numpy()

# Plug directly into MolPilot:
predictor = MyMPNNPredictor("my_model.pt")
suggestions = generate_analogs("c1ccccc1", predictor)
```

No base-class inheritance required — duck-typing is sufficient.

---

## Interpretability: SHAP Atom-Level Attribution

MolPilot maps fingerprint-level SHAP values back to atoms using RDKit's `bitInfo`
dictionary, which records which atom (and at what Morgan radius) contributed each
set bit.

```
bit_i is ON in mol M
    └─ bit_i was activated by atom a_j at radius r
        └─ SHAP(bit_i) ──accumulate──► atom_weight(a_j)
```

Per-atom weights are averaged over all bits that touch each atom, preventing
hub atoms from being artificially up-weighted. The resulting heatmap is rendered
as a green (activating) / red (suppressing) overlay on the 2-D structure.

---

## Activity Cliffs

A pair (A, B) is flagged as an **activity cliff** when:
- **Tanimoto(A, B) ≥ threshold** (default 0.70 — structurally similar)
- **|P(A) − P(B)| ≥ threshold** (default 1.0 log unit — property-dissimilar)

This is the canonical SAR signal that small structural changes can cause
disproportionate property changes. MolPilot surfaces these in the design table
so chemists can either exploit them (large gain from a small edit) or de-risk them
(synthesise the cliff pair early rather than assuming smooth interpolation).

```python
from mol_pilot import activity_cliff_score

result = activity_cliff_score(
    smiles_a="Clc1ccc(Cl)cc1",
    smiles_b="c1ccc(Cl)cc1",
    prop_a=-2.79,
    prop_b=0.84,
    sim_threshold=0.50,
    delta_threshold=1.0,
)
print(result.label)
# ⚠ Activity cliff: Tanimoto=0.58 >= 0.50, |ΔP|=3.63 >= 1.0
```

---

## Generative Design

Analogs are generated via two complementary routes:

1. **SMARTS transforms** — a curated library of 16 common medicinal-chemistry moves:
   halogen scan (ArH→F/Cl/Br), methyl/CF3/OMe addition, ring N-oxide, NH→NMe, etc.
2. **MMP fragmentation** — `rdMMPA.FragmentMol` cuts each rotatable bond and
   recombines the core with a library of 18 drug-like R-group replacements.

All generated analogs are:
- De-duplicated (canonical SMILES)
- Filtered by Tanimoto window (avoids near-identical or unrelated structures)
- Scored by: `Δpredicted_property + 0.5 × Tanimoto − 0.1 × SA_score`
- Ranked and returned as a pandas DataFrame

SA score uses the Ertl & Schuffenhauer algorithm (via `rdkit.Contrib.SA_Score`)
with an automatic fallback to a ring-complexity/stereocentre proxy.

---

## Cookbook Examples (Real Data, Real Output)

Run these in order from the repo root:

### CB-1: Train a solubility model on AqSolDB (TDC)
```bash
python cookbook/01_train_solubility_lgbm.py
```
Downloads 9,982 AqSolDB compounds from TDC, trains a LightGBM ECFP4 model,
evaluates on a hold-out set, and saves the model + kNN retriever.

**Actual results (from this run):**
```
Split sizes  — train: 7,984 | valid: 998 | test: 998
In-sample  R2: 0.9042  RMSE: 0.7355  MAE: 0.5090
Hold-out   R2: 0.7226  RMSE: 1.2199  MAE: 0.8849
```

### CB-2: MolPilot walkthrough on imatinib (Gleevec)
```bash
python cookbook/02_molpilot_imatinib_walkthrough.py
```
Demonstrates the full pipeline on imatinib (STI-571, first-in-class BCR-ABL inhibitor):
prediction → SHAP atom map → similar-compound retrieval → ranked design suggestions → CLI batch score.

**Actual prediction:**
```
Predicted logS: -2.791  → Moderate solubility class
MW: 493.6 Da  LogP: 4.59  QED: 0.389  Lipinski: PASS
Top analog (rank 1): Ethyl homolog (+0.20 logS, SA=2.42)
```

Outputs: `imatinib_shap.png`, `imatinib_design_suggestions.csv`, `demo_batch_predictions.csv`

### CB-3: Activity cliff detection
```bash
python cookbook/03_activity_cliff_demo.py
```
Analyses six documented matched pairs (Delaney solubility SAR), flags confirmed cliffs,
and screens 200 AqSolDB hold-out compounds for undiscovered cliffs.

**Actual results:**
```
Chlorobenzene vs 1,4-Dichlorobenzene  Tanimoto=0.583  ΔlogS=-3.63  CLIFF
Toluene vs 4-Nitrotoluene             Tanimoto=0.318  ΔlogS=-1.50  CLIFF
Aniline vs 4-Chloroaniline            Tanimoto=0.412  ΔlogS=-1.37  CLIFF
Hexane vs Hexanol                     Tanimoto=0.538  ΔlogS=+2.51  CLIFF
```

Output: `cliff_documented_pairs.csv`, `activity_cliff_grid.png`

All outputs are compatible with **DataWarrior** and **Benchling** table imports.

---

## Project Structure

```
mol_pilot/
  __init__.py        public API surface
  io.py              SMILES/SDF parsing, salt removal, standardisation (RDKit)
  featurize.py       ECFP fingerprints + RDKit descriptors; Featurizer protocol
  models.py          Predictor protocol; LightGBM + sklearn adapters; load/save
  interpret.py       SHAP bit→atom mapping; activity-cliff detection
  design.py          SMARTS transforms + MMP recombination; SA-scored ranking
  retrieve.py        Tanimoto kNN index over reference fingerprint library
  app.py             Streamlit UI (4 tabs: prediction, SHAP, retrieval, design)
  cli.py             `molpilot score / design / ui` commands
  _sa_score.py       SA score (Ertl via RDKit Contrib, proxy fallback)
examples/
  demo_ligands.smi   10 drug-like reference SMILES
cookbook/
  01_train_solubility_lgbm.py
  02_molpilot_imatinib_walkthrough.py
  03_activity_cliff_demo.py
  data/              generated outputs (predictions, images, CSVs)
tests/               62 pytest tests across all modules
```

---

## Running the Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
# 62 passed in ~4 s
```

---

## License

GNU GPL v3.0 — see [LICENSE](LICENSE).

---

## Contributing

Issues and pull requests welcome. Please run `ruff check` and `pytest` before submitting.
Feature suggestions most welcome in the [Issues](https://github.com/molpilot/molpilot/issues) tracker.

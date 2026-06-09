"""
MolPilot Streamlit UI.

Run:  streamlit run mol_pilot/app.py
  or: molpilot ui
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MolPilot",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _mol_to_png_bytes(mol: Chem.Mol, size: tuple[int, int] = (400, 300)) -> bytes:
    """Render a molecule to PNG bytes using RDKit Cairo drawer."""
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        from PIL import Image as PILImage
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _mol_to_highlighted_png(
    mol: Chem.Mol,
    atom_weights: np.ndarray,
    size: tuple[int, int] = (600, 400),
) -> bytes:
    """Render with per-atom SHAP colouring (green=activating, red=suppressing)."""
    max_w = np.abs(atom_weights).max() or 1.0
    norm_w = atom_weights / max_w

    atom_colors: dict[int, tuple[float, float, float]] = {}
    for i, w in enumerate(norm_w):
        if w > 0:
            intensity = float(w)
            atom_colors[i] = (0.2, 0.5 + 0.5 * intensity, 0.2)
        else:
            intensity = float(abs(w))
            atom_colors[i] = (0.5 + 0.5 * intensity, 0.2, 0.2)

    highlight_atoms = list(range(mol.GetNumAtoms()))
    highlight_bonds: list[int] = []

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=highlight_bonds,
            highlightBondColors={},
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        st.warning(f"Highlight rendering unavailable: {exc}")
        return _mol_to_png_bytes(mol, size)


def _qed_score(mol: Chem.Mol) -> float:
    from rdkit.Chem import QED
    return float(QED.qed(mol))


def _basic_props(mol: Chem.Mol) -> dict:
    return {
        "MW": round(Descriptors.MolWt(mol), 1),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "QED": round(_qed_score(mol), 3),
    }


# ── session state defaults ────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "predictor": None,
        "retriever": None,
        "last_smiles": "",
        "prediction_result": None,
        "shap_weights": None,
        "similar_df": None,
        "design_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    st.sidebar.image("https://raw.githubusercontent.com/rdkit/rdkit/master/Docs/Book/images/logo.png",
                     width=80, use_container_width=False)
    st.sidebar.title("🧪 MolPilot")
    st.sidebar.caption("Molecular Design Copilot v0.1")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Load Model")

    model_path = st.sidebar.text_input(
        "Model path (.joblib)",
        value=str(Path("models_saved/solubility_lgbm.joblib")),
        key="model_path_input",
    )
    retriever_path = st.sidebar.text_input(
        "Retriever path (.joblib)",
        value=str(Path("models_saved/solubility_retriever.joblib")),
        key="retriever_path_input",
    )

    if st.sidebar.button("Load", key="load_btn"):
        _load_model(model_path, retriever_path)

    if st.session_state.predictor is not None:
        st.sidebar.success(
            f"✓ Model: {st.session_state.predictor.model_name}"
        )
    else:
        st.sidebar.info("No model loaded — upload or train one first.")

    if st.session_state.retriever is not None:
        st.sidebar.success(
            f"✓ Retriever: {len(st.session_state.retriever)} compounds"
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Disclaimer:** Research tooling only. Not validated for clinical, "
        "regulatory, or commercial decision-making."
    )


def _load_model(model_path: str, retriever_path: str) -> None:
    from mol_pilot.models import load_predictor
    from mol_pilot.retrieve import NearestNeighborRetriever

    try:
        st.session_state.predictor = load_predictor(model_path)
        st.sidebar.success("Model loaded.")
    except Exception as exc:
        st.sidebar.error(f"Model load failed: {exc}")

    if Path(retriever_path).exists():
        try:
            st.session_state.retriever = NearestNeighborRetriever.load(retriever_path)
        except Exception as exc:
            st.sidebar.warning(f"Retriever load failed: {exc}")


# ── demo model (no file needed) ───────────────────────────────────────────────

def _load_demo_predictor() -> Any:
    """Build a trivially fast QED-based demo predictor (no training needed)."""
    from mol_pilot.featurize import ECFPFeaturizer

    class _QEDPredictor:
        featurizer = ECFPFeaturizer()
        model_name = "QED (demo)"
        task = "regression"

        def predict(self, smiles: list[str]) -> np.ndarray:
            vals = []
            for s in smiles:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    vals.append(float("nan"))
                else:
                    vals.append(_qed_score(mol))
            return np.array(vals)

        def save(self, path): pass

    return _QEDPredictor()


# ── main page ─────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    _sidebar()

    st.title("🧪 MolPilot — Molecular Design Copilot")
    st.caption(
        "Paste or draw a molecule to get instant property prediction, "
        "interpretability highlights, similar compounds, and ranked analog suggestions."
    )

    # ── input ──────────────────────────────────────────────────────────────
    col_input, col_preview = st.columns([3, 2])
    with col_input:
        smiles_input = st.text_input(
            "SMILES input",
            value=st.session_state.last_smiles or "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
            placeholder="Paste a SMILES string…",
            key="smiles_box",
        )
        run_col, demo_col = st.columns(2)
        run_btn = run_col.button("🔬 Analyse Molecule", type="primary", use_container_width=True)
        demo_btn = demo_col.button("⚡ Load Demo Predictor", use_container_width=True)

    if demo_btn:
        st.session_state.predictor = _load_demo_predictor()
        st.info("Demo QED predictor loaded (no file needed).")

    with col_preview:
        if smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                png = _mol_to_png_bytes(mol, size=(350, 260))
                st.image(png, caption="Query structure")
            else:
                st.warning("Could not parse SMILES.")

    if run_btn and smiles_input:
        if st.session_state.predictor is None:
            st.error("Load a model first (sidebar) or click **Load Demo Predictor**.")
        else:
            _run_analysis(smiles_input)

    # ── results tabs ───────────────────────────────────────────────────────
    if st.session_state.prediction_result is not None:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Prediction", "🎨 SHAP Highlights", "🔍 Similar Compounds", "⚗️ Design Suggestions"]
        )
        with tab1:
            _render_prediction_tab()
        with tab2:
            _render_shap_tab()
        with tab3:
            _render_similar_tab()
        with tab4:
            _render_design_tab()


# ── analysis runner ───────────────────────────────────────────────────────────

def _run_analysis(smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return

    predictor = st.session_state.predictor

    with st.spinner("Running analysis…"):
        # Prediction
        pred_val = float(predictor.predict([smiles])[0])

        # Basic physico-chemical props
        props = _basic_props(mol)

        st.session_state.prediction_result = {
            "smiles": smiles,
            "predicted_value": pred_val,
            "model_name": predictor.model_name,
            "props": props,
        }
        st.session_state.last_smiles = smiles

        # SHAP
        try:
            from mol_pilot.interpret import SHAPInterpreter
            interpreter = SHAPInterpreter(predictor)
            st.session_state.shap_weights = interpreter.atom_weights(smiles)
        except Exception as exc:
            st.session_state.shap_weights = None
            st.warning(f"SHAP computation skipped: {exc}")

        # Similar compounds
        if st.session_state.retriever is not None:
            try:
                st.session_state.similar_df = st.session_state.retriever.query(smiles, k=10)
            except Exception as exc:
                st.session_state.similar_df = None
                st.warning(f"Retrieval skipped: {exc}")
        else:
            st.session_state.similar_df = None

        # Design suggestions
        try:
            from mol_pilot.design import generate_analogs
            st.session_state.design_df = generate_analogs(
                smiles, predictor, n_analogs=25, include_mmp=True
            )
        except Exception as exc:
            st.session_state.design_df = None
            st.warning(f"Design generation skipped: {exc}")

    st.success("Analysis complete!")


# ── tab renderers ─────────────────────────────────────────────────────────────

def _render_prediction_tab() -> None:
    result = st.session_state.prediction_result
    if result is None:
        return

    st.subheader(f"Predicted {result['model_name']}")
    pred = result["predicted_value"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(result["model_name"], f"{pred:.3f}")
    props = result["props"]
    col2.metric("MW", f"{props['MW']} Da")
    col3.metric("LogP", props["LogP"])
    col4.metric("QED", props["QED"])

    st.markdown("---")
    st.subheader("Physicochemical Profile")

    props_display = {k: v for k, v in props.items() if k not in ("QED",)}
    lipinski = {
        "MW ≤ 500": props["MW"] <= 500,
        "LogP ≤ 5": props["LogP"] <= 5,
        "HBD ≤ 5": props["HBD"] <= 5,
        "HBA ≤ 10": props["HBA"] <= 10,
    }

    prop_col, lip_col = st.columns(2)
    with prop_col:
        df_props = pd.DataFrame(
            [{"Property": k, "Value": v} for k, v in props_display.items()]
        )
        st.dataframe(df_props, hide_index=True, use_container_width=True)

    with lip_col:
        st.markdown("**Lipinski Ro5 Check**")
        for rule, passed in lipinski.items():
            icon = "✅" if passed else "❌"
            st.markdown(f"{icon} {rule}")


def _render_shap_tab() -> None:
    result = st.session_state.prediction_result
    weights = st.session_state.shap_weights

    if result is None:
        st.info("Run analysis first.")
        return

    smiles = result["smiles"]
    mol = Chem.MolFromSmiles(smiles)

    st.subheader("SHAP Atom-Level Attribution")
    st.caption(
        "Green atoms activate the predicted property. Red atoms suppress it. "
        "Intensity reflects the magnitude of the contribution."
    )

    if weights is not None and mol is not None:
        img_bytes = _mol_to_highlighted_png(mol, weights, size=(700, 450))
        st.image(img_bytes, caption="SHAP-highlighted 2D structure", use_container_width=False)

        # Atom weight table
        st.markdown("**Per-atom SHAP weights**")
        atom_df = pd.DataFrame(
            {
                "Atom idx": list(range(len(weights))),
                "Symbol": [mol.GetAtomWithIdx(i).GetSymbol() for i in range(len(weights))],
                "SHAP weight": [round(float(w), 4) for w in weights],
                "Direction": ["activating" if w > 0 else "suppressing" for w in weights],
            }
        )
        atom_df = atom_df.sort_values("SHAP weight", key=abs, ascending=False)
        st.dataframe(atom_df, hide_index=True, use_container_width=True)
    else:
        st.warning("SHAP weights not available. Make sure the model supports TreeSHAP.")
        if mol:
            png = _mol_to_png_bytes(mol, size=(600, 400))
            st.image(png)


def _render_similar_tab() -> None:
    st.subheader("Nearest-Neighbour Similar Compounds")

    if st.session_state.similar_df is None:
        st.info(
            "No retriever index loaded. Build one from the cookbook or load "
            "a saved .joblib retriever via the sidebar."
        )
        return

    df = st.session_state.similar_df
    if df.empty:
        st.warning("No similar compounds found above the similarity threshold.")
        return

    # Similarity plot
    prop_cols = [c for c in df.columns if c not in ("rank", "smiles", "tanimoto")]
    if prop_cols:
        fig = px.scatter(
            df,
            x="tanimoto",
            y=prop_cols[0],
            hover_data=["smiles"],
            title=f"Tanimoto vs {prop_cols[0]}",
            labels={"tanimoto": "Tanimoto Similarity", prop_cols[0]: prop_cols[0]},
            color="tanimoto",
            color_continuous_scale="viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, hide_index=True, use_container_width=True)


def _render_design_tab() -> None:
    st.subheader("Ranked Analog Design Suggestions")
    st.caption(
        "Analogs ranked by predicted property improvement, structural similarity "
        "to the query, and synthetic accessibility (SA score 1–10)."
    )

    df = st.session_state.design_df

    if df is None or df.empty:
        st.warning("No design suggestions generated.")
        return

    # Filters
    fcol1, fcol2, fcol3 = st.columns(3)
    min_delta = fcol1.slider("Min Δ property", float(df.delta_property.min()),
                              float(df.delta_property.max()), 0.0, step=0.1)
    max_sa = fcol2.slider("Max SA score", 1.0, 10.0, 7.0, step=0.5)
    min_sim = fcol3.slider("Min Tanimoto", 0.0, 1.0, 0.3, step=0.05)

    filtered = df[
        (df.delta_property >= min_delta) &
        (df.sa_score <= max_sa) &
        (df.tanimoto_to_query >= min_sim)
    ]

    st.markdown(f"**{len(filtered)} suggestions** after filtering (from {len(df)} total)")

    # Scatter: delta_property vs sa_score
    if not filtered.empty:
        fig = px.scatter(
            filtered,
            x="sa_score",
            y="delta_property",
            hover_data=["smiles", "transform", "tanimoto_to_query"],
            color="delta_property",
            color_continuous_scale="RdYlGn",
            title="Design space: Δ property vs Synthetic Accessibility",
            labels={"sa_score": "SA Score (lower=easier)", "delta_property": "Δ Predicted Property"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    # Activity cliff check
    result = st.session_state.prediction_result
    if result and not filtered.empty:
        from mol_pilot.interpret import activity_cliff_score as acs
        query_val = result["predicted_value"]
        query_smi = result["smiles"]
        cliffs = []
        for _, row in filtered.head(10).iterrows():
            cliff = acs(query_smi, row.smiles, query_val, row.predicted_property,
                        sim_threshold=0.70, delta_threshold=1.0)
            if cliff.is_cliff:
                cliffs.append({"smiles": row.smiles, "tanimoto": cliff.tanimoto,
                                "delta": cliff.delta_property, "note": cliff.label})
        if cliffs:
            st.warning(f"⚠ **{len(cliffs)} activity cliff(s)** detected among top suggestions:")
            st.dataframe(pd.DataFrame(cliffs), hide_index=True, use_container_width=True)

    # Full suggestions table with download
    st.dataframe(filtered, hide_index=True, use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download CSV",
        data=csv_bytes,
        file_name="design_suggestions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

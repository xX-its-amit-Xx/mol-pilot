"""Tests for mol_pilot.models"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mol_pilot.featurize import ECFPFeaturizer
from mol_pilot.models import (
    LightGBMPredictor,
    Predictor,
    SklearnPredictor,
    evaluate_regression,
    load_predictor,
)

TRAIN_SMILES = [
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
TRAIN_Y = np.array([1.6, 0.3, -3.6, -1.3, 0.8, 1.4, 1.3, 0.5, 0.9, 0.8])


class TestLightGBMPredictor:
    def test_fit_predict_shape(self):
        pred = LightGBMPredictor()
        pred.fit(TRAIN_SMILES, TRAIN_Y)
        out = pred.predict(TRAIN_SMILES)
        assert out.shape == (len(TRAIN_SMILES),)

    def test_predict_single(self):
        pred = LightGBMPredictor()
        pred.fit(TRAIN_SMILES, TRAIN_Y)
        val = pred.predict(["c1ccccc1"])
        assert isinstance(val[0], float)

    def test_satisfies_protocol(self):
        pred = LightGBMPredictor()
        assert isinstance(pred, Predictor)

    def test_save_load_roundtrip(self):
        pred = LightGBMPredictor(model_name="test_logS")
        pred.fit(TRAIN_SMILES, TRAIN_Y)
        preds_before = pred.predict(TRAIN_SMILES)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            pred.save(path)
            loaded = LightGBMPredictor.load(path)

        preds_after = loaded.predict(TRAIN_SMILES)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

    def test_evaluate_regression(self):
        pred = LightGBMPredictor()
        pred.fit(TRAIN_SMILES, TRAIN_Y)
        metrics = evaluate_regression(pred, TRAIN_SMILES, TRAIN_Y)
        assert "r2" in metrics and "rmse" in metrics and "mae" in metrics
        assert metrics["rmse"] >= 0
        assert isinstance(metrics["r2"], float)


class TestSklearnPredictor:
    def test_fit_predict_shape(self):
        pred = SklearnPredictor()
        pred.fit(TRAIN_SMILES, TRAIN_Y)
        out = pred.predict(TRAIN_SMILES)
        assert out.shape == (len(TRAIN_SMILES),)

    def test_satisfies_protocol(self):
        pred = SklearnPredictor()
        assert isinstance(pred, Predictor)

    def test_save_load_roundtrip(self):
        pred = SklearnPredictor()
        pred.fit(TRAIN_SMILES, TRAIN_Y)
        preds_before = pred.predict(TRAIN_SMILES)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sklearn_model.joblib"
            pred.save(path)
            loaded = load_predictor(path)

        preds_after = loaded.predict(TRAIN_SMILES)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

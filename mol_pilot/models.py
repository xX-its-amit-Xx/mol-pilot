"""
Model-agnostic Predictor protocol + LightGBM and scikit-learn adapters.

Any object that implements ``predict(smiles: list[str]) -> np.ndarray``
satisfies the Predictor protocol and can be used anywhere MolPilot expects
a property model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import joblib
import numpy as np

from mol_pilot.featurize import ECFPFeaturizer, Featurizer

logger = logging.getLogger(__name__)


# ── Predictor protocol ────────────────────────────────────────────────────────

@runtime_checkable
class Predictor(Protocol):
    """Structural protocol — implement these two methods to plug any model in."""

    @property
    def featurizer(self) -> Featurizer:
        """The featurizer this predictor was trained with."""
        ...

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Return predicted property values (regression) or class probabilities."""
        ...


# ── LightGBM adapter ─────────────────────────────────────────────────────────

class LightGBMPredictor:
    """Wrap a LightGBM regressor/classifier behind the Predictor interface.

    Parameters
    ----------
    featurizer:
        Featurizer used to convert SMILES to feature vectors.
    model:
        A fitted ``lightgbm.LGBMRegressor`` / ``LGBMClassifier`` instance, or
        ``None`` to create a default regressor.
    task:
        ``"regression"`` (default) or ``"classification"``.
    """

    def __init__(
        self,
        featurizer: Featurizer | None = None,
        model: Any | None = None,
        task: str = "regression",
        model_name: str = "property",
    ) -> None:
        self.featurizer = featurizer or ECFPFeaturizer()
        self.task = task
        self.model_name = model_name

        if model is not None:
            self.model = model
        else:
            import lightgbm as lgb
            if task == "regression":
                self.model = lgb.LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=63,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )

    # ── training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        smiles_val: list[str] | None = None,
        y_val: np.ndarray | None = None,
        **lgb_fit_kwargs: Any,
    ) -> "LightGBMPredictor":
        X = self.featurizer.transform(smiles)
        feature_names = self.featurizer.get_feature_names()
        fit_kwargs: dict[str, Any] = dict(lgb_fit_kwargs)
        # Pass feature names via fit() — LightGBM ≥ 4.x removed the feature_name_ setter
        fit_kwargs.setdefault("feature_name", feature_names)
        if smiles_val is not None and y_val is not None:
            X_val = self.featurizer.transform(smiles_val)
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model.fit(X, y, **fit_kwargs)
        logger.info("Trained %s on %d molecules", self.model_name, len(smiles))
        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, smiles: list[str]) -> np.ndarray:
        X = self.featurizer.transform(smiles)
        return self.model.predict(X)

    def predict_proba(self, smiles: list[str]) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model does not support predict_proba (use a classifier).")
        X = self.featurizer.transform(smiles)
        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "type": "LightGBMPredictor",
                "featurizer": self.featurizer,
                "model": self.model,
                "task": self.task,
                "model_name": self.model_name,
            },
            path,
        )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMPredictor":
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.featurizer = data["featurizer"]
        instance.model = data["model"]
        instance.task = data["task"]
        instance.model_name = data["model_name"]
        return instance

    def __repr__(self) -> str:
        return (
            f"LightGBMPredictor(name={self.model_name!r}, task={self.task!r}, "
            f"featurizer={self.featurizer!r})"
        )


# ── scikit-learn adapter ──────────────────────────────────────────────────────

class SklearnPredictor:
    """Wrap any scikit-learn estimator behind the Predictor interface."""

    def __init__(
        self,
        featurizer: Featurizer | None = None,
        model: Any | None = None,
        model_name: str = "property",
    ) -> None:
        self.featurizer = featurizer or ECFPFeaturizer()
        self.model_name = model_name
        if model is not None:
            self.model = model
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)

    def fit(self, smiles: list[str], y: np.ndarray) -> "SklearnPredictor":
        X = self.featurizer.transform(smiles)
        self.model.fit(X, y)
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        X = self.featurizer.transform(smiles)
        return self.model.predict(X)

    def predict_proba(self, smiles: list[str]) -> np.ndarray:
        X = self.featurizer.transform(smiles)
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"type": "SklearnPredictor", "featurizer": self.featurizer, "model": self.model,
             "model_name": self.model_name},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SklearnPredictor":
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.featurizer = data["featurizer"]
        instance.model = data["model"]
        instance.model_name = data["model_name"]
        return instance

    def __repr__(self) -> str:
        return f"SklearnPredictor(name={self.model_name!r}, model={self.model!r})"


# ── convenience load/save ─────────────────────────────────────────────────────

def load_predictor(path: str | Path) -> LightGBMPredictor | SklearnPredictor:
    """Load a saved predictor from disk; detects type automatically."""
    data = joblib.load(path)
    ptype = data.get("type", "LightGBMPredictor")
    if ptype == "LightGBMPredictor":
        return LightGBMPredictor.load(path)
    elif ptype == "SklearnPredictor":
        return SklearnPredictor.load(path)
    raise ValueError(f"Unknown predictor type stored in {path}: {ptype!r}")


def save_predictor(predictor: Any, path: str | Path) -> None:
    """Save any predictor that has a .save() method."""
    if hasattr(predictor, "save"):
        predictor.save(path)
    else:
        raise TypeError(f"{type(predictor).__name__} does not implement .save()")


# ── quick model evaluation helpers ───────────────────────────────────────────

def evaluate_regression(
    predictor: Any,
    smiles_test: list[str],
    y_test: np.ndarray,
) -> dict[str, float]:
    """Return R², RMSE, and MAE for a regression predictor on a hold-out set."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = predictor.predict(smiles_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }

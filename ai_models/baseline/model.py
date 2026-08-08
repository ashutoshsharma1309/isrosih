"""
Candidate model definitions and the trained-model artifact format.

The artifact ("bundle") saved by train.py is a joblib dict containing the
fitted estimator plus everything inference needs to be self-contained:
feature names/medians, label names, metrics, and version metadata.
`BaselineRainfallModel` adapts a bundle to the project-wide RainfallModel
interface (ai_models/base.py) for backend integration in Phase 7.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_models.base import ModelOutput, RainfallModel
from ai_models.baseline.config import BaselineConfig

logger = logging.getLogger(__name__)

BUNDLE_FILENAME_TEMPLATE = "rainfall_model_{version}.pkl"

#: Sklearn-style estimators that accept sample_weight in fit().
_NEEDS_SAMPLE_WEIGHT = {"gradient_boosting", "xgboost"}


def build_candidate_models(seed: int) -> dict[str, Any]:
    """The four baseline candidates compared in Phase 3.

    Class imbalance (extreme events are rare) is handled per model:
    class_weight='balanced' where supported, balanced sample weights
    otherwise (see train.fit_model).
    """
    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    except ImportError as exc:  # e.g. missing libomp on macOS
        logger.warning("XGBoost unavailable (%s) — comparing the three sklearn models only", exc)
    return models


def needs_sample_weight(name: str) -> bool:
    return name in _NEEDS_SAMPLE_WEIGHT


# ---------------------------------------------------------------------
# Artifact (bundle) I/O
# ---------------------------------------------------------------------
def save_bundle(
    *,
    config: BaselineConfig,
    model_name: str,
    estimator: Any,
    feature_names: list[str],
    feature_medians: dict[str, float],
    metrics: dict[str, Any],
    dataset_info: dict[str, Any],
) -> Path:
    bundle = {
        "version": config.model_version,
        "model_name": model_name,
        "estimator": estimator,
        "feature_names": feature_names,
        "feature_medians": feature_medians,
        "label_names": config.label_names,
        "metrics": metrics,
        "dataset_info": dataset_info,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    config.saved_models_dir.mkdir(parents=True, exist_ok=True)
    path = config.saved_models_dir / BUNDLE_FILENAME_TEMPLATE.format(version=config.model_version)
    joblib.dump(bundle, path)
    logger.info("Saved model bundle: %s (%s)", path, model_name)
    return path


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m ai_models.baseline.train` first."
        )
    bundle = joblib.load(path)
    required = {"estimator", "feature_names", "label_names", "version"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model bundle {path} is missing keys: {sorted(missing)}")
    return bundle


# ---------------------------------------------------------------------
# Project-wide interface adapter
# ---------------------------------------------------------------------
class BaselineRainfallModel(RainfallModel):
    """Adapts a trained bundle to the RainfallModel interface consumed by
    the backend's PredictionService (integration lands in Phase 7)."""

    _RISK_BY_CATEGORY = {0: "low", 1: "heavy", 2: "extreme"}

    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None

    def load(self, artifact_dir: Path) -> None:
        version = BaselineConfig().model_version
        self._bundle = load_bundle(
            Path(artifact_dir) / BUNDLE_FILENAME_TEMPLATE.format(version=version)
        )
        self.version = self._bundle["version"]

    def predict(self, features: dict[str, float] | None, image: np.ndarray | None) -> ModelOutput:
        if self._bundle is None:
            raise RuntimeError("Model not loaded — call load() first")
        if features is None:
            raise ValueError("The baseline model requires tabular features")

        row = pd.DataFrame(
            [{name: features.get(name, self._bundle["feature_medians"].get(name)) for name in self._bundle["feature_names"]}]
        )
        probabilities = self._bundle["estimator"].predict_proba(row)[0]
        category = int(np.argmax(probabilities))
        return ModelOutput(
            probability=float(probabilities[1:].sum()),  # P(high-impact: heavy or extreme)
            risk_level=self._RISK_BY_CATEGORY[category],
            confidence=float(probabilities.max()),
            model_version=f"baseline-{self._bundle['model_name']}-{self._bundle['version']}",
            explanation_context={
                "feature_vector": row.iloc[0].to_dict(),
                "class_probabilities": {
                    self._bundle["label_names"][i]: float(p) for i, p in enumerate(probabilities)
                },
            },
        )

    def feature_names(self) -> list[str]:
        if self._bundle is None:
            raise RuntimeError("Model not loaded — call load() first")
        return list(self._bundle["feature_names"])

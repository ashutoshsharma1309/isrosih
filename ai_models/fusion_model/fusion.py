"""
The three fusion strategies compared in Phase 5.

Every strategy exposes the same surface — `fit(...)` then
`predict_proba(frame) -> (n_samples, num_classes)` — so train.py can
compare them without special-casing, and predict.py can serve whichever
one won without knowing which it is.

Approach 1 — WeightedFusion (late / decision-level)
    Convex blend of the two upstream probability vectors. The weather
    weight is swept on the validation split; nothing is learned beyond
    that single scalar, which makes it the honest floor the trained
    approaches must beat.

Approach 2 — FeatureFusion (feature / intermediate-level)
    Weather features + satellite scene statistics + both upstream risk
    scores concatenated into one vector, then a Random Forest or XGBoost
    classifier on top.

Approach 3 — NeuralFusion (joint representation)
    Separate MLP encoders per modality, concatenated into a fusion layer.
    On a dataset this small it is the most likely to overfit; it is
    included because the phase brief asks for the comparison, and the
    result is reported as measured.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ai_models.fusion_model.config import FusionConfig
from ai_models.fusion_model.utils import align_probabilities

logger = logging.getLogger(__name__)


class FusionStrategy(Protocol):
    """Common surface every fusion approach implements."""

    name: str
    approach: str

    def fit(self, frame: pd.DataFrame, target: pd.Series, config: FusionConfig) -> "FusionStrategy": ...

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray: ...


# ---------------------------------------------------------------------
# Approach 1 — weighted late fusion
# ---------------------------------------------------------------------
class WeightedFusion:
    """Convex blend of the upstream probability vectors.

    `final = w * weather_probabilities + (1 - w) * satellite_probabilities`

    The weight is chosen on validation data by the caller via
    `fit_weight`; `fit` itself is a no-op so the class satisfies the
    shared strategy surface.
    """

    approach = "weighted"

    def __init__(self, weather_weight: float = 0.5) -> None:
        if not 0.0 <= weather_weight <= 1.0:
            raise ValueError(f"weather_weight must be in [0, 1], got {weather_weight}")
        self.weather_weight = float(weather_weight)
        self.name = f"weighted_fusion_w{self.weather_weight:.2f}"
        self._config: FusionConfig | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series, config: FusionConfig) -> "WeightedFusion":
        self._config = config
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        config = self._config or FusionConfig()
        weather = frame[config.weather_probability_columns()].to_numpy(dtype=np.float64)
        satellite = frame[config.satellite_probability_columns()].to_numpy(dtype=np.float64)
        blended = self.weather_weight * weather + (1.0 - self.weather_weight) * satellite
        return blended / blended.sum(axis=1, keepdims=True)


def fit_weight(
    frame: pd.DataFrame, target: pd.Series, config: FusionConfig, score_fn
) -> tuple[WeightedFusion, list[dict[str, float]]]:
    """Sweep the weather weight on validation data; keep the best.

    Args:
        score_fn: callable(y_true, y_pred) -> float, maximised over the grid.

    Returns the fitted strategy and the full sweep for the report.
    """
    sweep: list[dict[str, float]] = []
    best_weight, best_score = 0.5, -np.inf
    for weight in np.arange(0.0, 1.0 + 1e-9, config.weight_grid_step):
        candidate = WeightedFusion(float(weight)).fit(frame, target, config)
        predictions = candidate.predict_proba(frame).argmax(axis=1)
        score = float(score_fn(target, predictions))
        sweep.append({"weather_weight": round(float(weight), 2), "score": round(score, 4)})
        if score > best_score:
            best_weight, best_score = float(weight), score
    logger.info(
        "Weighted fusion: best weather weight %.2f (validation %s %.3f)",
        best_weight, config.selection_metric, best_score,
    )
    return WeightedFusion(best_weight).fit(frame, target, config), sweep


# ---------------------------------------------------------------------
# Approach 2 — feature-level fusion
# ---------------------------------------------------------------------
class FeatureFusion:
    """Concatenated feature vector → a single tree-ensemble classifier."""

    approach = "feature_level"

    def __init__(self, estimator_name: str, seed: int) -> None:
        self.name = estimator_name
        self.seed = seed
        self.estimator: Any = _build_estimator(estimator_name, seed)
        self._features: list[str] = []

    def fit(self, frame: pd.DataFrame, target: pd.Series, config: FusionConfig) -> "FeatureFusion":
        self._features = config.feature_columns()
        # Inverse-frequency weighting: a missed heavy-rain day is the
        # expensive error, and the classes are heavily imbalanced.
        weights = compute_sample_weight("balanced", target)
        self.estimator.fit(frame[self._features], target, sample_weight=weights)
        self._classes = np.asarray(self.estimator.classes_, dtype=int)
        self._num_classes = config.num_classes
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return align_probabilities(
            self.estimator.predict_proba(frame[self._features]), self._classes, self._num_classes
        )


def _build_estimator(name: str, seed: int) -> Any:
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            tree_method="hist",
            eval_metric="mlogloss",
        )
    raise ValueError(f"Unknown feature-fusion estimator: {name}")


# ---------------------------------------------------------------------
# Approach 3 — neural fusion network
# ---------------------------------------------------------------------
class NeuralFusion:
    """Per-modality MLP encoders → fusion layer → classifier head.

    Weather encoder and satellite encoder learn separate representations
    before they meet, so the network can weigh the modalities per sample
    instead of applying one global weight the way Approach 1 does.
    """

    approach = "neural"
    name = "neural_fusion"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._model: Any = None
        self._scaler_weather = StandardScaler()
        self._scaler_satellite = StandardScaler()
        self._weather_columns: list[str] = []
        self._satellite_columns: list[str] = []
        self._num_classes = 3

    def fit(self, frame: pd.DataFrame, target: pd.Series, config: FusionConfig) -> "NeuralFusion":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self._weather_columns = config.weather_feature_columns()
        self._satellite_columns = config.satellite_feature_columns()
        self._num_classes = config.num_classes

        weather = self._scaler_weather.fit_transform(frame[self._weather_columns])
        satellite = self._scaler_satellite.fit_transform(frame[self._satellite_columns])
        y = torch.tensor(target.to_numpy(), dtype=torch.long)

        self._model = _FusionNet(
            weather_dim=weather.shape[1],
            satellite_dim=satellite.shape[1],
            config=config,
        )
        # Class-weighted loss, matching the imbalance handling elsewhere.
        counts = np.bincount(target.to_numpy(), minlength=config.num_classes).astype(np.float64)
        weights = np.divide(
            counts.sum(), counts * config.num_classes, out=np.zeros_like(counts), where=counts > 0
        )
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=config.nn_learning_rate, weight_decay=config.nn_weight_decay
        )

        weather_t = torch.tensor(weather, dtype=torch.float32)
        satellite_t = torch.tensor(satellite, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(weather_t, satellite_t, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.nn_batch_size, shuffle=True)

        self._model.train()
        for epoch in range(1, config.nn_epochs + 1):
            losses = []
            for batch_weather, batch_satellite, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(batch_weather, batch_satellite), batch_y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            if epoch % 20 == 0 or epoch == 1:
                logger.info("[neural_fusion] epoch %d/%d loss %.4f",
                            epoch, config.nn_epochs, float(np.mean(losses)))
        self._model.eval()
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("NeuralFusion is not fitted — call fit() first")
        weather = torch.tensor(
            self._scaler_weather.transform(frame[self._weather_columns]), dtype=torch.float32
        )
        satellite = torch.tensor(
            self._scaler_satellite.transform(frame[self._satellite_columns]), dtype=torch.float32
        )
        with torch.no_grad():
            logits = self._model(weather, satellite)
            return torch.softmax(logits, dim=1).numpy().astype(np.float64)


def _FusionNet(*, weather_dim: int, satellite_dim: int, config: FusionConfig):
    """Build the two-encoder fusion network (kept lazy so torch stays optional)."""
    import torch
    from torch import nn

    class FusionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weather_encoder = nn.Sequential(
                nn.Linear(weather_dim, config.nn_hidden_weather),
                nn.ReLU(),
                nn.Dropout(config.nn_dropout),
            )
            self.satellite_encoder = nn.Sequential(
                nn.Linear(satellite_dim, config.nn_hidden_satellite),
                nn.ReLU(),
                nn.Dropout(config.nn_dropout),
            )
            self.fusion = nn.Sequential(
                nn.Linear(config.nn_hidden_weather + config.nn_hidden_satellite, config.nn_hidden_fusion),
                nn.ReLU(),
                nn.Dropout(config.nn_dropout),
            )
            self.head = nn.Linear(config.nn_hidden_fusion, config.num_classes)

        def forward(self, weather: "torch.Tensor", satellite: "torch.Tensor") -> "torch.Tensor":
            joined = torch.cat([self.weather_encoder(weather), self.satellite_encoder(satellite)], dim=1)
            return self.head(self.fusion(joined))

    return FusionNet()


# ---------------------------------------------------------------------
# Single-modality references (for the improvement measurement)
# ---------------------------------------------------------------------
class SingleModality:
    """Passes an upstream model's probabilities through unchanged.

    Used as the baseline the fusion must beat — the "weather model alone"
    and "satellite model alone" rows of the hybrid report.
    """

    approach = "single_modality"

    def __init__(self, modality: str) -> None:
        if modality not in {"weather", "satellite"}:
            raise ValueError(f"modality must be 'weather' or 'satellite', got {modality!r}")
        self.modality = modality
        self.name = f"{modality}_only"
        self._columns: list[str] = []

    def fit(self, frame: pd.DataFrame, target: pd.Series, config: FusionConfig) -> "SingleModality":
        self._columns = (
            config.weather_probability_columns()
            if self.modality == "weather"
            else config.satellite_probability_columns()
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self._columns].to_numpy(dtype=np.float64)

"""
Final hybrid prediction pipeline.

    Data Input
        ↓
    Weather Model (Phase 3)  ──┐
                               ├── Fusion Engine (Phase 5) → Final Prediction
    Satellite Model (Phase 4) ─┘

The predictor accepts raw inputs and drives the upstream models itself,
so a caller only has to supply what it observed. Anything it cannot
observe may be passed pre-computed instead; anything still missing falls
back to the training median and is reported in `imputed_features` rather
than being silently invented.

CLI:
    backend/.venv/bin/python -m ai_models.fusion_model.predict --input '{
        "weather_data": {"temperature_c": 29.4, "humidity_pct": 91.0,
                         "pressure_hpa": 1002.0, "wind_speed_ms": 7.1,
                         "wind_direction_deg": 250.0, "cloud_cover_pct": 96.0,
                         "rain_sum_1d": 48.0, "rain_sum_3d": 130.0},
        "satellite_image_path": "data/raw/satellite/.../kerala_2018-08-15.jpg",
        "location": {"region": "Kerala", "latitude": 10.0, "longitude": 76.3},
        "timestamp": "2018-08-15T05:15:00Z"}'
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ai_models.base import ModelOutput, RainfallModel
from ai_models.fusion_model.config import FUSION_BUNDLE_TEMPLATE, REPO_ROOT, FusionConfig
from ai_models.fusion_model.utils import (
    confidence_label,
    model_agreement,
    risk_level,
    risk_score,
)

logger = logging.getLogger(__name__)


class InvalidInputError(ValueError):
    """Raised when the prediction payload is malformed."""


class ModelNotTrainedError(RuntimeError):
    """Raised when the fusion bundle has not been trained yet."""


class HybridPredictor:
    """Loads the fusion bundle once and serves end-to-end predictions."""

    def __init__(self, model_path: Path | None = None, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        path = model_path or (
            self.config.saved_models_dir
            / FUSION_BUNDLE_TEMPLATE.format(version=self.config.model_version)
        )
        if not Path(path).exists():
            raise ModelNotTrainedError(
                f"No fusion model at {path}. Run `python -m ai_models.fusion_model.train` first."
            )
        self.bundle = joblib.load(path)
        self.strategy = self.bundle["strategy"]
        self.matcher = self.bundle["matcher"]
        self._weather_model: Any = None
        self._satellite_model: Any = None
        self._satellite_transform: Any = None

    # ------------------------------------------------------------------
    # Upstream branches
    # ------------------------------------------------------------------
    def _weather_probabilities(self, weather: dict[str, Any]) -> np.ndarray | None:
        """Run the Phase 3 tabular model on the supplied conditions."""
        from ai_models.baseline.config import BaselineConfig
        from ai_models.baseline.model import BUNDLE_FILENAME_TEMPLATE, load_bundle
        from ai_models.fusion_model.utils import align_probabilities

        if self._weather_model is None:
            baseline_config = BaselineConfig()
            path = self.config.saved_models_dir / BUNDLE_FILENAME_TEMPLATE.format(
                version=baseline_config.model_version
            )
            if not path.exists():
                logger.warning("Phase 3 model unavailable at %s", path)
                return None
            self._weather_model = load_bundle(path)

        bundle = self._weather_model
        medians = bundle.get("feature_medians", {})
        row = pd.DataFrame(
            [{name: float(weather.get(name, medians.get(name, 0.0))) for name in bundle["feature_names"]}]
        )
        return align_probabilities(
            bundle["estimator"].predict_proba(row),
            bundle["estimator"].classes_,
            self.config.num_classes,
        )

    def _satellite_probabilities(self, image_path: Path) -> np.ndarray | None:
        """Run the Phase 4 vision model on a scene."""
        import torch

        from ai_models.satellite_model.config import SatelliteConfig
        from ai_models.satellite_model.model import CHECKPOINT_TEMPLATE, load_checkpoint
        from ai_models.satellite_model.preprocessing import build_transforms, load_rgb

        if self._satellite_model is None:
            satellite_config = SatelliteConfig()
            path = self.config.saved_models_dir / CHECKPOINT_TEMPLATE.format(
                version=satellite_config.model_version
            )
            if not path.exists():
                logger.warning("Phase 4 checkpoint unavailable at %s", path)
                return None
            self._satellite_model, _ = load_checkpoint(path, device="cpu")
            self._satellite_transform = build_transforms(satellite_config, training=False)

        image = load_rgb(image_path)
        with torch.no_grad():
            tensor = self._satellite_transform(image).unsqueeze(0)
            return torch.softmax(self._satellite_model(tensor), dim=1).numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        weather, satellite, location, timestamp, image_path = self._validate(payload)
        notes: list[str] = []

        # --- Branch 1: weather -----------------------------------------
        weather_probabilities = _as_vector(weather.get("class_probabilities"), self.config.num_classes)
        if weather_probabilities is None:
            weather_probabilities = self._weather_probabilities(weather)
        if weather_probabilities is None and weather.get("weather_risk_score") is not None:
            weather_probabilities = _vector_from_score(float(weather["weather_risk_score"]))
            notes.append("weather branch reconstructed from a scalar risk score")
        if weather_probabilities is None:
            raise InvalidInputError(
                "Cannot evaluate the weather branch: supply weather_data, a weather_risk_score, "
                "or train the Phase 3 model."
            )

        # --- Branch 2: satellite ---------------------------------------
        satellite_probabilities = _as_vector(satellite.get("class_probabilities"), self.config.num_classes)
        if satellite_probabilities is None and image_path is not None:
            satellite_probabilities = self._satellite_probabilities(image_path)
        if satellite_probabilities is None and satellite.get("satellite_risk_score") is not None:
            satellite_probabilities = _vector_from_score(float(satellite["satellite_risk_score"]))
            notes.append("satellite branch reconstructed from a scalar risk score")
        if satellite_probabilities is None:
            raise InvalidInputError(
                "Cannot evaluate the satellite branch: supply satellite_image_path, "
                "satellite_features with class_probabilities, or a satellite_risk_score."
            )

        weather_score = float(risk_score(weather_probabilities)[0])
        satellite_score = float(risk_score(satellite_probabilities)[0])

        # --- Scene features (extracted from the image when available) ---
        if image_path is not None and not _has_scene_features(satellite, self.config):
            satellite = {**satellite, **self._extract_scene_features(payload, image_path)}

        # --- Fusion input row -------------------------------------------
        row, imputed = self._build_row(
            weather, satellite, timestamp, weather_score, satellite_score,
            weather_probabilities[0], satellite_probabilities[0],
        )

        fused = self.strategy.predict_proba(row)
        probabilities = np.asarray(fused, dtype=np.float64)[0]
        category = int(np.argmax(probabilities))
        probability = float(probabilities[1:].sum())

        # --- Historical analogues + confidence ---------------------------
        # Replaying a day that is itself a reference event would match it at
        # 100% and inflate confidence; drop that self-match.
        self_key = (
            (str(location.get("region")), timestamp[:10])
            if location.get("region") and timestamp
            else None
        )
        matches = self._match_history(row, self_key)
        confidence_pct, breakdown = self._confidence(
            float(probabilities.max()), weather_score, satellite_score, matches
        )

        return {
            "event_prediction": self.config.event_names[category],
            "risk_probability": round(probability, 4),
            "confidence": confidence_label(confidence_pct),
            "confidence_pct": round(confidence_pct, 1),
            "risk_level": risk_level(probability, self.config),
            "predicted_category": category,
            "class_probabilities": {
                self.config.label_names[i]: round(float(p), 4) for i, p in enumerate(probabilities)
            },
            "contributing_models": {
                "weather_risk_score": round(weather_score, 4),
                "satellite_risk_score": round(satellite_score, 4),
                "agreement": round(breakdown["agreement"], 4) if breakdown["agreement"] is not None else None,
            },
            "confidence_breakdown": breakdown,
            "similar_historical_events": matches,
            "location": location,
            "timestamp": timestamp,
            "imputed_features": imputed,
            "notes": notes,
            "model": f"{self.bundle['model_name']}-{self.bundle['version']}",
            "approach": self.bundle["approach"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    def _extract_scene_features(self, payload: dict[str, Any], image_path: Path) -> dict[str, Any]:
        from ai_models.satellite_model.preprocessing import ImageLoadError, extract_scene_features

        def optional(key: str) -> Path | None:
            value = payload.get(key)
            if not value:
                return None
            path = Path(value)
            return path if path.is_absolute() else REPO_ROOT / path

        try:
            return extract_scene_features(image_path, optional("ir_image_path"), optional("previous_image_path"))
        except ImageLoadError as exc:
            logger.warning("Scene features unavailable: %s", exc)
            return {}

    def _build_row(
        self,
        weather: dict[str, Any],
        satellite: dict[str, Any],
        timestamp: str | None,
        weather_score: float,
        satellite_score: float,
        weather_probabilities: np.ndarray,
        satellite_probabilities: np.ndarray,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Assemble the fusion feature row, recording every imputed value."""
        medians = self.bundle.get("feature_medians", {})
        supplied: dict[str, Any] = {**weather, **satellite}
        supplied["weather_risk_score"] = weather_score
        supplied["satellite_risk_score"] = satellite_score

        if timestamp and ("season_sin" not in supplied or "season_cos" not in supplied):
            moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            day_of_year = moment.timetuple().tm_yday
            supplied.setdefault("season_sin", math.sin(2 * math.pi * day_of_year / 365.25))
            supplied.setdefault("season_cos", math.cos(2 * math.pi * day_of_year / 365.25))

        values: dict[str, float] = {}
        imputed: list[str] = []
        for name in self.bundle["feature_names"]:
            value = supplied.get(name)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = medians.get(name, 0.0)
                imputed.append(name)
            values[name] = float(value)

        # The weighted strategy consumes the upstream vectors directly.
        for index in range(self.config.num_classes):
            values[f"weather_prob_{index}"] = float(weather_probabilities[index])
            values[f"satellite_prob_{index}"] = float(satellite_probabilities[index])

        if imputed:
            logger.info("Imputed %d missing feature(s) from training medians: %s", len(imputed), imputed)
        return pd.DataFrame([values]), imputed

    def _match_history(
        self, row: pd.DataFrame, exclude: tuple[str, str] | None = None
    ) -> list[dict[str, Any]]:
        try:
            return self.matcher.match(row.iloc[0].to_dict(), self.config, top_k=3, exclude=exclude)
        except Exception as exc:
            logger.warning("Historical matching unavailable: %s", exc)
            return []

    def _confidence(
        self,
        max_probability: float,
        weather_score: float,
        satellite_score: float,
        matches: list[dict[str, Any]],
    ) -> tuple[float, dict[str, Any]]:
        """Blend prediction strength, branch agreement, and historical support.

        Components that cannot be computed are dropped and the remaining
        weights renormalised, so an absent signal never counts as zero.
        """
        agreement = model_agreement(weather_score, satellite_score)
        similarity = float(matches[0]["similarity"]) if matches else None

        components = [
            (max_probability, self.config.confidence_weight_probability),
            (agreement, self.config.confidence_weight_agreement),
            (similarity, self.config.confidence_weight_similarity),
        ]
        available = [(value, weight) for value, weight in components if value is not None]
        total_weight = sum(weight for _, weight in available)
        score = sum(value * weight for value, weight in available) / total_weight if total_weight else 0.0

        return score * 100.0, {
            "prediction_probability": round(max_probability, 4),
            "agreement": agreement if agreement is None else round(agreement, 4),
            "historical_similarity": similarity if similarity is None else round(similarity, 4),
            "weights": {
                "prediction_probability": self.config.confidence_weight_probability,
                "agreement": self.config.confidence_weight_agreement,
                "historical_similarity": self.config.confidence_weight_similarity,
            },
        }

    # ------------------------------------------------------------------
    def _validate(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None, Path | None]:
        if not isinstance(payload, dict):
            raise InvalidInputError("Input must be a JSON object")

        weather = payload.get("weather_data") or {}
        satellite = payload.get("satellite_features") or {}
        if not isinstance(weather, dict) or not isinstance(satellite, dict):
            raise InvalidInputError("weather_data and satellite_features must be objects")

        location = payload.get("location") or {}
        if isinstance(location, str):
            location = {"region": location}
        if not isinstance(location, dict):
            raise InvalidInputError("location must be an object or a region name")

        for field, low, high in (("latitude", -90.0, 90.0), ("longitude", -180.0, 180.0)):
            if location.get(field) is not None:
                try:
                    value = float(location[field])
                except (TypeError, ValueError) as exc:
                    raise InvalidInputError(f"location.{field} is not numeric") from exc
                if not low <= value <= high:
                    raise InvalidInputError(f"location.{field}={value} outside [{low}, {high}]")

        timestamp = payload.get("timestamp")
        if timestamp:
            try:
                datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            except ValueError as exc:
                raise InvalidInputError(f"Unparseable timestamp: {timestamp!r}") from exc

        image_path: Path | None = None
        if payload.get("satellite_image_path"):
            candidate = Path(str(payload["satellite_image_path"]))
            image_path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
            if not image_path.is_file():
                raise InvalidInputError(f"Satellite image does not exist: {image_path}")

        return weather, satellite, location, (str(timestamp) if timestamp else None), image_path


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _as_vector(value: Any, num_classes: int) -> np.ndarray | None:
    """Accept a list or a {label: probability} mapping as a probability vector."""
    if value is None:
        return None
    if isinstance(value, dict):
        ordered = [float(v) for _, v in sorted(value.items(), key=lambda kv: str(kv[0]))]
    elif isinstance(value, (list, tuple)):
        ordered = [float(v) for v in value]
    else:
        return None
    if len(ordered) != num_classes:
        raise InvalidInputError(f"class_probabilities must have {num_classes} entries, got {len(ordered)}")
    vector = np.asarray([ordered], dtype=np.float64)
    total = vector.sum()
    if total <= 0:
        raise InvalidInputError("class_probabilities must sum to a positive value")
    return vector / total


def _vector_from_score(score: float) -> np.ndarray:
    """Reconstruct a class vector from a scalar risk score.

    The split between heavy and extreme is unknowable from one number, so
    the whole risk mass is assigned to heavy. Callers are told via `notes`.
    """
    score = float(np.clip(score, 0.0, 1.0))
    return np.asarray([[1.0 - score, score, 0.0]], dtype=np.float64)


def _has_scene_features(satellite: dict[str, Any], config: FusionConfig) -> bool:
    return any(satellite.get(name) is not None for name in config.satellite_features)


# ---------------------------------------------------------------------
# Project-wide interface adapter (consumed by Phase 7)
# ---------------------------------------------------------------------
class HybridRainfallModel(RainfallModel):
    """Adapts the fusion bundle to the shared `RainfallModel` contract."""

    _RISK_BY_CATEGORY = {0: "low", 1: "heavy", 2: "extreme"}

    def __init__(self) -> None:
        self._predictor: HybridPredictor | None = None

    def load(self, artifact_dir: Path) -> None:
        config = FusionConfig()
        self._predictor = HybridPredictor(
            model_path=Path(artifact_dir) / FUSION_BUNDLE_TEMPLATE.format(version=config.model_version),
            config=config,
        )
        self.version = self._predictor.bundle["version"]

    def predict(self, features: dict[str, float] | None, image: np.ndarray | None) -> ModelOutput:
        if self._predictor is None:
            raise RuntimeError("Model not loaded — call load() first")
        if features is None:
            raise ValueError("The hybrid model requires weather features")

        result = self._predictor.predict({"weather_data": features, "satellite_features": features})
        return ModelOutput(
            probability=result["risk_probability"],
            risk_level=self._RISK_BY_CATEGORY[result["predicted_category"]],
            confidence=result["confidence_pct"] / 100.0,
            model_version=f"fusion-{result['model']}",
            explanation_context={
                "class_probabilities": result["class_probabilities"],
                "contributing_models": result["contributing_models"],
                "similar_historical_events": result["similar_historical_events"],
            },
        )

    def feature_names(self) -> list[str]:
        if self._predictor is None:
            raise RuntimeError("Model not loaded — call load() first")
        return list(self._predictor.bundle["feature_names"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VARUNA AI hybrid rainfall prediction")
    parser.add_argument("--input", required=True, help="JSON object (see module docstring)")
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    try:
        payload = json.loads(args.input)
        result = HybridPredictor(model_path=args.model).predict(payload)
    except (InvalidInputError, ModelNotTrainedError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

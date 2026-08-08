"""
Prediction module — structured inference on the trained baseline model.

Input: current weather conditions + location (plus optional recent
rainfall history and date). Output: a structured dict ready for API
integration in Phase 7.

Honesty rules:
- Raises ModelNotTrainedError when no trained bundle exists.
- Rejects invalid inputs (InvalidInputError) instead of guessing.
- Historical features not supplied are filled with training medians, and
  the response says so (`assumed_features`) with confidence downgraded —
  the caller always knows what the model actually saw.

CLI:
    backend/.venv/bin/python -m ai_models.baseline.predict \
        --input '{"latitude": 10.0, "longitude": 76.3, "temperature_c": 26,
                  "humidity_pct": 92, "pressure_hpa": 1001, "wind_speed_ms": 9,
                  "cloud_cover_pct": 95}'
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

import numpy as np
import pandas as pd

from ai_models.baseline.config import BaselineConfig
from ai_models.baseline.model import BUNDLE_FILENAME_TEMPLATE, load_bundle

logger = logging.getLogger(__name__)


class InvalidInputError(ValueError):
    """Raised when prediction input is missing fields or out of range."""


class ModelNotTrainedError(RuntimeError):
    """Raised when no trained model bundle is available."""


#: field → (min, max, required)
_INPUT_SPEC: dict[str, tuple[float, float, bool]] = {
    "latitude": (-90.0, 90.0, True),
    "longitude": (-180.0, 180.0, True),
    "temperature_c": (-60.0, 60.0, True),
    "humidity_pct": (0.0, 100.0, True),
    "pressure_hpa": (850.0, 1100.0, True),
    "wind_speed_ms": (0.0, 120.0, True),
    "cloud_cover_pct": (0.0, 100.0, True),
    "wind_direction_deg": (0.0, 360.0, False),
    "recent_rainfall_mm_1d": (0.0, 2000.0, False),
    "recent_rainfall_mm_3d": (0.0, 5000.0, False),
    "recent_rainfall_mm_7d": (0.0, 10000.0, False),
    "recent_rainfall_mm_30d": (0.0, 30000.0, False),
}

_HISTORY_TO_FEATURE = {
    "recent_rainfall_mm_1d": "rain_sum_1d",
    "recent_rainfall_mm_3d": "rain_sum_3d",
    "recent_rainfall_mm_7d": "rain_sum_7d",
    "recent_rainfall_mm_30d": "rain_sum_30d",
}


class RainfallPredictor:
    """Loads a trained bundle once and serves structured predictions."""

    def __init__(self, model_path: Path | None = None, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        path = model_path or (
            self.config.saved_models_dir
            / BUNDLE_FILENAME_TEMPLATE.format(version=self.config.model_version)
        )
        try:
            self.bundle = load_bundle(path)
        except FileNotFoundError as exc:
            raise ModelNotTrainedError(str(exc)) from exc
        logger.info(
            "Loaded %s (%s, trained %s)",
            path.name, self.bundle["model_name"], self.bundle.get("trained_at", "unknown"),
        )

    # ------------------------------------------------------------------
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = self._validate(payload)
        features, assumed = self._build_feature_row(values, payload.get("date"))

        probabilities = self.bundle["estimator"].predict_proba(features)[0]
        category = int(np.argmax(probabilities))
        max_probability = float(probabilities.max())
        confidence = self._confidence(max_probability, downgraded=bool(assumed))

        return {
            "prediction": self.bundle["label_names"][category],
            "category": category,
            "risk_score": round(float(probabilities[1:].sum()), 4),  # P(heavy or extreme)
            "class_probabilities": {
                self.bundle["label_names"][i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            },
            "confidence": confidence,
            "model": f"{self.bundle['model_name']}-{self.bundle['version']}",
            "assumed_features": assumed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    def _validate(self, payload: dict[str, Any]) -> dict[str, float]:
        if not isinstance(payload, dict):
            raise InvalidInputError("Input must be a JSON object")
        values: dict[str, float] = {}
        for field, (lo, hi, required) in _INPUT_SPEC.items():
            if field not in payload or payload[field] is None:
                if required:
                    raise InvalidInputError(f"Missing required field: {field}")
                continue
            try:
                value = float(payload[field])
            except (TypeError, ValueError) as exc:
                raise InvalidInputError(f"Field {field} is not numeric: {payload[field]!r}") from exc
            if math.isnan(value) or not (lo <= value <= hi):
                raise InvalidInputError(f"Field {field}={value} outside valid range [{lo}, {hi}]")
            values[field] = value
        return values

    def _build_feature_row(
        self, values: dict[str, float], date: str | None
    ) -> tuple[pd.DataFrame, list[str]]:
        medians: dict[str, float] = self.bundle.get("feature_medians", {})
        assumed: list[str] = []
        row: dict[str, float] = {}

        for name in self.bundle["feature_names"]:
            if name in values:
                row[name] = values[name]
            elif name in ("season_sin", "season_cos"):
                row[name] = self._season(name, date)
            elif name == "rain_trend_3d":
                row[name] = self._trend(values, medians, assumed)
            else:
                history_key = next(
                    (k for k, v in _HISTORY_TO_FEATURE.items() if v == name), None
                )
                if history_key and history_key in values:
                    row[name] = values[history_key]
                else:
                    row[name] = float(medians.get(name, 0.0))
                    assumed.append(name)
        return pd.DataFrame([row], columns=self.bundle["feature_names"]), assumed

    @staticmethod
    def _season(component: str, date: str | None) -> float:
        try:
            when = datetime.fromisoformat(date) if date else datetime.now(timezone.utc)
        except ValueError as exc:
            raise InvalidInputError(f"Unparseable date: {date!r}") from exc
        angle = 2 * math.pi * when.timetuple().tm_yday / 365.25
        return math.sin(angle) if component == "season_sin" else math.cos(angle)

    @staticmethod
    def _trend(values: dict[str, float], medians: dict[str, float], assumed: list[str]) -> float:
        # Without a full history the caller can't supply the trend directly;
        # approximate from provided sums when possible, else fall back.
        if "recent_rainfall_mm_3d" in values and "recent_rainfall_mm_7d" in values:
            previous_window = max(values["recent_rainfall_mm_7d"] - values["recent_rainfall_mm_3d"], 0.0)
            return values["recent_rainfall_mm_3d"] - previous_window * (3.0 / 4.0)
        assumed.append("rain_trend_3d")
        return float(medians.get("rain_trend_3d", 0.0))

    @staticmethod
    def _confidence(max_probability: float, *, downgraded: bool) -> str:
        levels = ["Low", "Medium", "High"]
        index = 2 if max_probability >= 0.75 else 1 if max_probability >= 0.5 else 0
        if downgraded and index > 0:
            index -= 1  # assumed features → honest confidence downgrade
        return levels[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VARUNA AI baseline rainfall prediction")
    parser.add_argument("--input", required=True, help="JSON object with weather conditions")
    parser.add_argument("--model", type=Path, default=None, help="Path to a model bundle")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    try:
        payload = json.loads(args.input)
        result = RainfallPredictor(model_path=args.model).predict(payload)
    except (InvalidInputError, ModelNotTrainedError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

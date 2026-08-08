"""
Satellite prediction module — structured inference on a trained checkpoint.

Input (dict): satellite_image_path (required), timestamp, latitude,
longitude, plus optional ir_image_path / previous_image_path for the
interpretable feature block.

Output: structured dict compatible with the Phase 3 tabular predictor —
same confidence vocabulary (High/Medium/Low), same risk-score semantics
(P(heavy) + P(extreme)) — so the Phase 5 fusion can consume both
uniformly.

CLI:
    backend/.venv/bin/python -m ai_models.satellite_model.predict \
        --input '{"satellite_image_path": "data/raw/satellite/.../kerala_2018-08-15.jpg",
                  "timestamp": "2018-08-15T05:15:00Z", "latitude": 10.0, "longitude": 76.3}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ai_models.satellite_model.config import REPO_ROOT, SatelliteConfig, resolve_device
from ai_models.satellite_model.model import CHECKPOINT_TEMPLATE, load_checkpoint
from ai_models.satellite_model.preprocessing import (
    ImageLoadError,
    build_transforms,
    extract_scene_features,
    load_rgb,
)

logger = logging.getLogger(__name__)


class InvalidInputError(ValueError):
    """Raised when prediction input is malformed."""


class ModelNotTrainedError(RuntimeError):
    """Raised when no trained satellite checkpoint exists."""


_PATTERN_BY_CATEGORY = {0: "Low Risk", 1: "High Risk", 2: "Extreme Risk"}


class SatellitePredictor:
    """Loads the trained checkpoint once and serves scene predictions."""

    def __init__(self, model_path: Path | None = None, config: SatelliteConfig | None = None) -> None:
        self.config = config or SatelliteConfig()
        self.device = resolve_device()
        path = model_path or (
            self.config.saved_models_dir
            / CHECKPOINT_TEMPLATE.format(version=self.config.model_version)
        )
        try:
            self.model, self.meta = load_checkpoint(path, self.device)
        except FileNotFoundError as exc:
            raise ModelNotTrainedError(str(exc)) from exc
        self.transform = build_transforms(self.config, training=False)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_path, ir_path, previous_path = self._validate(payload)

        image = load_rgb(image_path)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        category = int(np.argmax(probabilities))
        max_probability = float(probabilities.max())

        features: dict[str, Any] | None = None
        try:
            features = extract_scene_features(image_path, ir_path, previous_path)
        except ImageLoadError as exc:
            logger.warning("Scene features unavailable: %s", exc)

        return {
            "satellite_risk_score": round(float(probabilities[1:].sum()), 4),
            "cloud_pattern": _PATTERN_BY_CATEGORY[category],
            "cloud_condition": self.meta["label_names"][category],
            "category": category,
            "class_probabilities": {
                self.meta["label_names"][i]: round(float(p), 4) for i, p in enumerate(probabilities)
            },
            "confidence": "High" if max_probability >= 0.75 else "Medium" if max_probability >= 0.5 else "Low",
            "scene_features": features,
            "model": f"{self.meta['model_name']}-{self.meta['version']}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    def _validate(self, payload: dict[str, Any]) -> tuple[Path, Path | None, Path | None]:
        if not isinstance(payload, dict):
            raise InvalidInputError("Input must be a JSON object")
        raw_path = payload.get("satellite_image_path")
        if not raw_path or not isinstance(raw_path, str):
            raise InvalidInputError("Missing required field: satellite_image_path")
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        if not image_path.is_file():
            raise InvalidInputError(f"Satellite image does not exist: {image_path}")

        for field, lo, hi in (("latitude", -90.0, 90.0), ("longitude", -180.0, 180.0)):
            if field in payload and payload[field] is not None:
                try:
                    value = float(payload[field])
                except (TypeError, ValueError) as exc:
                    raise InvalidInputError(f"Field {field} is not numeric") from exc
                if not (lo <= value <= hi):
                    raise InvalidInputError(f"Field {field}={value} outside [{lo}, {hi}]")

        if payload.get("timestamp"):
            try:
                datetime.fromisoformat(str(payload["timestamp"]).replace("Z", "+00:00"))
            except ValueError as exc:
                raise InvalidInputError(f"Unparseable timestamp: {payload['timestamp']!r}") from exc

        def optional_path(key: str) -> Path | None:
            value = payload.get(key)
            if not value:
                return None
            path = Path(value)
            return path if path.is_absolute() else REPO_ROOT / path

        return image_path, optional_path("ir_image_path"), optional_path("previous_image_path")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VARUNA AI satellite scene prediction")
    parser.add_argument("--input", required=True, help="JSON object (see module docstring)")
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    try:
        payload = json.loads(args.input)
        result = SatellitePredictor(model_path=args.model).predict(payload)
    except (InvalidInputError, ModelNotTrainedError, ImageLoadError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

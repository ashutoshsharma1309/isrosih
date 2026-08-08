"""
Shared helpers for the fusion package.

Kept deliberately small: probability-vector alignment (upstream estimators
may not have seen every class), risk-level vocabulary, seeding, and the
logging setup shared by the fusion entry points.
"""

from __future__ import annotations

import logging
import random
from typing import Sequence

import numpy as np

from ai_models.fusion_model.config import FusionConfig

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Consistent log format across the fusion entry points."""
    logging.basicConfig(
        level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )


def set_seeds(seed: int) -> None:
    """Seed every RNG the fusion pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch is optional for the sklearn-only paths
        logger.debug("torch unavailable — skipping its seed")


def align_probabilities(
    probabilities: np.ndarray, classes: Sequence[int], num_classes: int
) -> np.ndarray:
    """Expand an estimator's probability matrix to the full class space.

    A classifier trained on a split that never saw class 2 returns two
    columns; downstream code always expects `num_classes`. Missing classes
    get probability 0 rather than a fabricated value.

    Args:
        probabilities: (n_samples, len(classes)) from `predict_proba`.
        classes: the estimator's `classes_`, in column order.
        num_classes: full class-space size.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(f"Expected a 2-D probability matrix, got shape {probabilities.shape}")
    if probabilities.shape[1] == num_classes and list(classes) == list(range(num_classes)):
        return probabilities

    expanded = np.zeros((probabilities.shape[0], num_classes), dtype=np.float64)
    for column, class_id in enumerate(classes):
        index = int(class_id)
        if 0 <= index < num_classes:
            expanded[:, index] = probabilities[:, column]
        else:
            logger.warning("Estimator emitted unknown class %s — dropped", class_id)
    return expanded


def risk_score(probabilities: np.ndarray) -> np.ndarray:
    """P(high-impact) = P(heavy) + P(extreme) — the project-wide risk score."""
    matrix = np.atleast_2d(np.asarray(probabilities, dtype=np.float64))
    return matrix[:, 1:].sum(axis=1)


def risk_level(score: float, config: FusionConfig) -> str:
    """Map a risk score to the operational escalation vocabulary."""
    if score <= config.risk_low_max:
        return "LOW"
    if score <= config.risk_moderate_max:
        return "MODERATE"
    if score <= config.risk_high_max:
        return "HIGH"
    return "CRITICAL"


def confidence_label(percentage: float) -> str:
    """High/Medium/Low vocabulary shared with the Phase 3 and 4 predictors."""
    if percentage >= 75.0:
        return "High"
    if percentage >= 50.0:
        return "Medium"
    return "Low"


def model_agreement(weather_score: float | None, satellite_score: float | None) -> float | None:
    """Agreement between the two branches in [0, 1]; None if a branch is absent.

    Both scores answer "how likely is high-impact rain"; when they land in
    the same place the hybrid answer is better supported.
    """
    if weather_score is None or satellite_score is None:
        return None
    return float(1.0 - abs(float(weather_score) - float(satellite_score)))



"""
Abstract interfaces every VARUNA AI model must implement.

These contracts decouple the serving layer (backend/) from model
internals: the backend only ever sees `RainfallModel`, so tabular (Phase 3),
vision (Phase 4), and hybrid (Phase 5) models are interchangeable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelOutput:
    """Standardised prediction result shared by all model families."""

    probability: float  # P(high-impact rain event) in [0, 1]
    risk_level: str  # low | moderate | heavy | extreme
    confidence: float  # calibrated confidence in [0, 1]
    model_version: str
    # Raw materials the explainability layer needs (e.g. feature vector,
    # preprocessed image tensor, layer activations).
    explanation_context: dict[str, Any] = field(default_factory=dict)


class RainfallModel(ABC):
    """Contract for all rainfall prediction models."""

    #: Semantic version of the trained artifact, set on load.
    version: str = "unversioned"

    @abstractmethod
    def load(self, artifact_dir: Path) -> None:
        """Load trained weights/artifacts from disk. Called once at startup."""

    @abstractmethod
    def predict(self, features: dict[str, float] | None, image: np.ndarray | None) -> ModelOutput:
        """Run inference.

        Args:
            features: meteorological feature dict (tabular models).
            image: preprocessed satellite image array (vision models).
            Hybrid models consume both; unused inputs may be None.
        """

    @abstractmethod
    def feature_names(self) -> list[str]:
        """Ordered feature names — required by the SHAP explainer."""

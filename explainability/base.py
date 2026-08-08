"""
Explainability layer contracts.

Every prediction served by VARUNA AI must be explainable. This package
turns model outputs into human-understandable evidence:

- SHAP feature attributions for tabular/meteorological inputs (Phase 6)
- Grad-CAM heatmaps for satellite-image inputs (Phase 6)
- A plain-language narrative synthesised from both

Implementations arrive in Phase 6; the interface is fixed now so models
built in Phases 3-5 expose the right hooks (see
ai_models.base.ModelOutput.explanation_context).
"""

from abc import ABC, abstractmethod
from typing import Any

from ai_models.base import ModelOutput


class Explainer(ABC):
    """Contract for all explanation backends (SHAP, Grad-CAM, ...)."""

    @abstractmethod
    def explain(self, output: ModelOutput) -> dict[str, Any]:
        """Produce an explanation payload conforming to the API schema
        (backend/app/schemas/prediction.py::Explanation)."""

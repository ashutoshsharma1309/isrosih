"""
Prediction service — the seam between the web API and the AI packages.

The API layer never imports model code directly; it goes through this
service. In Phase 5 this service will load the trained hybrid model
(from ai_models/) and the explainer (from explainability/) and delegate
to them. Keeping the boundary here means the API contract is stable
while models evolve.
"""

from app.core.config import settings
from app.schemas.prediction import PredictionRequest, PredictionResponse


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained model artifact is available to serve."""


class PredictionService:
    def __init__(self) -> None:
        # Phase 5: load model artifacts from settings.MODEL_ARTIFACT_DIR here.
        self._model = None

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if self._model is None:
            raise ModelNotAvailableError(
                "No trained rainfall model is deployed yet. Model integration "
                "is scheduled for Phases 3-6; this endpoint intentionally does "
                "not return fabricated predictions."
            )
        raise NotImplementedError  # replaced by real inference in Phase 5

    def model_info(self) -> dict:
        return {
            "model_loaded": self._model is not None,
            "artifact_dir": settings.MODEL_ARTIFACT_DIR,
            "expected_models": {
                "tabular": "Gradient-boosted classifier on meteorological features (Phase 3)",
                "vision": "CNN on INSAT satellite imagery (Phase 4)",
                "hybrid": "Fusion of tabular + vision outputs (Phase 5)",
            },
        }

"""
Unified explanation generator.

Assembles the four evidence sources into the single XAI payload the API
will serve in Phase 7:

    SHAP (numeric drivers)  ─┐
    Grad-CAM (scene regions) ├─► unified explanation ─► narrative
    Historical analogues    ─┤
    Confidence components   ─┘

Design rule throughout: **the payload reports what the model did, not what
would read well.** Where a branch carried no weight, where a value was
imputed, or where a class could not be predicted, the payload says so.
That is why `caveats` is a first-class field rather than an afterthought —
an explanation that hides its own limits is worse than none, because it
invites the operator to trust it further than the evidence allows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from explainability.explanation_generator import templates
from explainability.gradcam_explainer.config import GradCamConfig
from explainability.gradcam_explainer.gradcam import (
    GradCamExplanation,
    GradCamNotAvailableError,
    SatelliteGradCam,
)
from explainability.historical_explainer import (
    HistoricalExplainer,
    HistoricalExplanation,
    HistoricalExplanationUnavailableError,
)
from explainability.shap_explainer.config import ShapConfig
from explainability.shap_explainer.explainer import (
    FusionShapExplainer,
    ShapExplanation,
    load_fusion_bundle,
)

logger = logging.getLogger(__name__)


class ExplanationError(RuntimeError):
    """Raised when an explanation cannot be produced at all."""


@dataclass
class ConfidenceExplanation:
    """The confidence score decomposed into its measured components."""

    confidence_pct: float
    label: str
    model_agreement: float | None
    historical_similarity: float | None
    data_quality: str
    data_quality_score: float
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_pct": round(float(self.confidence_pct), 1),
            "confidence": self.label,
            "factors": {
                "model_agreement": (
                    None if self.model_agreement is None else round(float(self.model_agreement), 4)
                ),
                "historical_similarity": (
                    None
                    if self.historical_similarity is None
                    else round(float(self.historical_similarity), 4)
                ),
                "data_quality": self.data_quality,
                "data_quality_score": round(float(self.data_quality_score), 4),
            },
            "components": self.components,
        }


@dataclass
class UnifiedExplanation:
    """The complete XAI response for one prediction."""

    prediction: str
    probability: float
    risk_level: str
    confidence: ConfidenceExplanation
    shap: ShapExplanation
    gradcam: GradCamExplanation | None
    history: HistoricalExplanation | None
    narrative: str
    caveats: list[str]
    artifacts: dict[str, str] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, top_k: int = 6) -> dict[str, Any]:
        """The Phase 6 XAI response format."""
        top = self.shap.top(top_k)
        return {
            "prediction": self.prediction,
            "probability": round(float(self.probability), 4),
            "confidence": self.confidence.label,
            "confidence_pct": round(float(self.confidence.confidence_pct), 1),
            "risk_level": self.risk_level,
            "top_features": [c.label for c in top],
            "satellite_regions": (
                [r.name for r in self.gradcam.regions] if self.gradcam else []
            ),
            "explanation": self.narrative,
            "feature_attributions": [c.to_dict() for c in top],
            "confidence_explanation": self.confidence.to_dict(),
            "satellite_explanation": self.gradcam.to_dict() if self.gradcam else None,
            "historical_explanation": self.history.to_dict() if self.history else None,
            "shap_detail": self.shap.to_dict(top_k),
            "caveats": self.caveats,
            "artifacts": self.artifacts,
            "context": self.context,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


class ExplanationGenerator:
    """Produces the unified explanation for one region-day."""

    def __init__(
        self,
        bundle: dict[str, Any] | None = None,
        shap_config: ShapConfig | None = None,
        gradcam_config: GradCamConfig | None = None,
    ) -> None:
        self.shap_config = shap_config or ShapConfig()
        self.gradcam_config = gradcam_config or GradCamConfig()
        self.bundle = bundle if bundle is not None else load_fusion_bundle(self.shap_config)
        self.shap_explainer = FusionShapExplainer(self.bundle, self.shap_config)
        self._gradcam: SatelliteGradCam | None = None
        try:
            self.history_explainer: HistoricalExplainer | None = HistoricalExplainer(self.bundle)
        except HistoricalExplanationUnavailableError as exc:
            logger.warning("Historical explanations unavailable: %s", exc)
            self.history_explainer = None

    # ------------------------------------------------------------------
    def explain(
        self,
        row: pd.DataFrame,
        *,
        image_path: str | Path | None = None,
        render: bool = False,
        exclude_self: tuple[str, str] | None = None,
    ) -> UnifiedExplanation:
        """Explain one region-day.

        Args:
            row: single-row frame with the fusion features and upstream
                probability columns.
            image_path: True Color scene, if the imagery should be explained.
            render: also write the SHAP and Grad-CAM figures to `reports/`.
            exclude_self: (region, date) to drop from historical matching.
        """
        shap_explanation = self.shap_explainer.explain(row)
        caveats: list[str] = list(shap_explanation.notes)

        # Disclose attribution the narrative deliberately does not voice.
        silent = sum(
            abs(c.share)
            for c in shap_explanation.contributions
            if c.feature in self.shap_config.non_narratable_features
        )
        if silent >= 0.10:
            caveats.append(
                f"Location and calendar terms carry {silent * 100:.0f}% of the attribution for "
                "this prediction. They are real model inputs and appear in the attribution "
                "table, but they are not voiced as causes because they describe where and when "
                "the region is, not what the weather is doing."
            )

        gradcam_explanation = self._explain_scene(image_path, caveats)
        history = self._explain_history(row, exclude_self, caveats)

        confidence = self._explain_confidence(shap_explanation, gradcam_explanation, history, row)
        probability = float(shap_explanation.predicted_risk)
        risk_level = self._risk_level(probability)
        prediction = self._event_name(shap_explanation, gradcam_explanation)

        narrative = self._narrate(
            prediction, probability, risk_level, shap_explanation,
            gradcam_explanation, history, confidence,
        )
        for source in (gradcam_explanation, history):
            if source is not None:
                caveats.extend(getattr(source, "notes", []))

        artifacts: dict[str, str] = {}
        if render:
            artifacts = self._render(shap_explanation, gradcam_explanation, image_path)

        return UnifiedExplanation(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            shap=shap_explanation,
            gradcam=gradcam_explanation,
            history=history,
            narrative=narrative,
            caveats=_dedupe(caveats),
            artifacts=artifacts,
            context={
                "region": _cell(row, "region"),
                "date": _cell(row, "date"),
                "fusion_model": self.bundle.get("model_name"),
                "fusion_approach": self.bundle.get("approach"),
            },
        )

    # ------------------------------------------------------------------
    def _explain_scene(
        self, image_path: str | Path | None, caveats: list[str]
    ) -> GradCamExplanation | None:
        if image_path is None:
            return None
        if self._gradcam is None:
            self._gradcam = SatelliteGradCam(self.gradcam_config)
        try:
            return self._gradcam.explain(image_path)
        except (GradCamNotAvailableError, ValueError) as exc:
            logger.warning("Grad-CAM unavailable: %s", exc)
            caveats.append(f"No satellite explanation was produced: {exc}")
            return None

    def _explain_history(
        self, row: pd.DataFrame, exclude: tuple[str, str] | None, caveats: list[str]
    ) -> HistoricalExplanation | None:
        if self.history_explainer is None:
            return None
        from ai_models.fusion_model.config import FusionConfig

        try:
            return self.history_explainer.explain(row, FusionConfig(), exclude=exclude)
        except (HistoricalExplanationUnavailableError, ValueError) as exc:
            logger.warning("Historical explanation unavailable: %s", exc)
            caveats.append(f"No historical comparison was produced: {exc}")
            return None

    # ------------------------------------------------------------------
    def _explain_confidence(
        self,
        shap_explanation: ShapExplanation,
        gradcam: GradCamExplanation | None,
        history: HistoricalExplanation | None,
        row: pd.DataFrame,
    ) -> ConfidenceExplanation:
        """Rebuild the Phase 5 confidence blend, component by component.

        Data quality is measured, not asserted: it combines how much of the
        satellite scene was actually imaged with how many model inputs had
        to be filled from training medians.
        """
        from ai_models.fusion_model.config import FusionConfig
        from ai_models.fusion_model.utils import confidence_label, model_agreement

        config = FusionConfig()
        weather_risk = shap_explanation.branch_weights.get("weather", 0.0)
        weather_value = _cell(row, "weather_risk_score")
        satellite_value = _cell(row, "satellite_risk_score")
        agreement = model_agreement(weather_value, satellite_value)
        similarity = history.best_similarity if history else None

        valid_fraction = _cell(row, "valid_fraction")
        imputed = sum(
            len(entry.get("inputs", "").split(","))
            for entry in shap_explanation.unattributed_inputs
            if "median" in entry.get("reason", "")
        )
        quality_score = float(valid_fraction if valid_fraction is not None else 1.0)
        quality_score *= max(0.0, 1.0 - 0.05 * imputed)
        quality = (
            "Good" if quality_score >= 0.85 else "Fair" if quality_score >= 0.6 else "Poor"
        )

        parts = [
            (float(max(shap_explanation.predicted_risk, 1 - shap_explanation.predicted_risk)),
             config.confidence_weight_probability),
            (agreement, config.confidence_weight_agreement),
            (similarity, config.confidence_weight_similarity),
        ]
        available = [(value, weight) for value, weight in parts if value is not None]
        total_weight = sum(weight for _, weight in available)
        score = (
            sum(value * weight for value, weight in available) / total_weight
            if total_weight
            else 0.0
        )
        percentage = score * 100.0

        return ConfidenceExplanation(
            confidence_pct=percentage,
            label=confidence_label(percentage),
            model_agreement=agreement,
            historical_similarity=similarity,
            data_quality=quality,
            data_quality_score=quality_score,
            components={
                "prediction_strength": round(parts[0][0], 4),
                "weights": {
                    "prediction_strength": config.confidence_weight_probability,
                    "model_agreement": config.confidence_weight_agreement,
                    "historical_similarity": config.confidence_weight_similarity,
                },
                "usable_scene_fraction": valid_fraction,
                "imputed_input_count": imputed,
                "weather_branch_weight": weather_risk,
            },
        )

    # ------------------------------------------------------------------
    def _narrate(
        self,
        prediction: str,
        probability: float,
        risk_level: str,
        shap_explanation: ShapExplanation,
        gradcam: GradCamExplanation | None,
        history: HistoricalExplanation | None,
        confidence: ConfidenceExplanation,
    ) -> str:
        # Two kinds of driver are kept out of prose but stay in the
        # attribution table: median-filled inputs (describing them asserts
        # something never observed) and geometry/calendar terms ("elevated
        # longitude" is not a cause anyone can act on). How much attribution
        # they absorbed is disclosed by the caller.
        top = [
            c
            for c in shap_explanation.top(self.shap_config.top_k)
            if not c.imputed and c.feature not in self.shap_config.non_narratable_features
        ]
        raising = [
            templates.describe_driver(c.feature, c.label, True)
            for c in top if c.contribution > 0
        ][:3]
        lowering = [
            templates.describe_driver(c.feature, c.label, False)
            for c in top if c.contribution < 0
        ][:2]

        sentences = [
            templates.headline(prediction, probability, risk_level),
            templates.drivers_sentence(raising, lowering),
        ]
        if gradcam is not None:
            sentences.append(
                templates.satellite_sentence(
                    [r.to_dict() for r in gradcam.regions],
                    gradcam.coverage_label,
                    shap_explanation.branch_weights.get("satellite", 0.0),
                    gradcam.satellite_risk,
                )
            )
        if history is not None:
            sentences.append(templates.historical_sentence(history.best))
        sentences.append(
            templates.confidence_sentence(
                confidence.confidence_pct,
                confidence.model_agreement,
                confidence.historical_similarity,
                confidence.data_quality,
            )
        )
        return " ".join(s for s in sentences if s)

    # ------------------------------------------------------------------
    def _render(
        self,
        shap_explanation: ShapExplanation,
        gradcam: GradCamExplanation | None,
        image_path: str | Path | None,
    ) -> dict[str, str]:
        from explainability.gradcam_explainer.visualization import save_overlay
        from explainability.shap_explainer.visualization import (
            plot_feature_importance,
            plot_waterfall,
        )

        artifacts = {
            "shap_bar": str(plot_feature_importance(shap_explanation, self.shap_config)),
            "shap_waterfall": str(plot_waterfall(shap_explanation, self.shap_config)),
        }
        if gradcam is not None and image_path is not None:
            artifacts["gradcam_overlay"] = str(
                save_overlay(image_path, gradcam, self.gradcam_config)
            )
        return artifacts

    # ------------------------------------------------------------------
    @staticmethod
    def _risk_level(probability: float) -> str:
        from ai_models.fusion_model.config import FusionConfig
        from ai_models.fusion_model.utils import risk_level

        return risk_level(probability, FusionConfig())

    def _event_name(
        self, shap_explanation: ShapExplanation, gradcam: GradCamExplanation | None
    ) -> str:
        """Name the predicted event from the fused risk, not the branch models."""
        from ai_models.fusion_model.config import FusionConfig

        config = FusionConfig()
        risk = shap_explanation.predicted_risk
        if risk < 0.5:
            return config.event_names[0]
        # Extreme is only claimed when the satellite branch is weighted and
        # actually favours it; the fused model cannot otherwise distinguish.
        if (
            gradcam is not None
            and gradcam.predicted_category == 2
            and shap_explanation.branch_weights.get("satellite", 0.0) > 0
        ):
            return config.event_names[2]
        return config.event_names[1]


def _cell(row: pd.DataFrame, name: str) -> Any:
    if name not in row.columns:
        return None
    value = row.iloc[0][name]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value) if isinstance(value, (int, float)) else value


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered

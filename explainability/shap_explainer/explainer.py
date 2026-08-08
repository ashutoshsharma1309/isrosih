"""
SHAP attribution for the Phase 5 hybrid model.

The fusion engine is a *pipeline*, not a single estimator, so there is no
one model to hand to SHAP. The final risk is

    risk = w · P_weather(weather features) + (1 - w) · P_satellite(scene)

which means an input's effect on the decision runs through whichever
branch consumes it. This module therefore attributes by the chain rule:
SHAP on the branch that actually reads the input, scaled by that branch's
exact fusion weight.

Why this matters concretely: the shipped v1 fusion is `WeightedFusion`
with `w = 1.00`. Perturbing humidity, pressure or cloud density changes
its output by exactly 0.000000 — the blender reads only the two upstream
probability vectors. Running SHAP against the blender would honestly
return zero for every meteorological feature, and any non-zero number
printed next to "Humidity" would be fabricated. Attribution therefore
targets the Phase 3 estimator (which does read humidity) and is then
scaled by `w`.

Three strategies are supported, selected automatically:

| Selected fusion | Attribution path | SHAP method |
|---|---|---|
| `WeightedFusion` | per-branch, scaled by the branch weight | exact TreeSHAP on the Phase 3 forest |
| `FeatureFusion` | directly over the 22-feature fusion vector | exact TreeSHAP on the fusion tree ensemble |
| `NeuralFusion` | directly over the 22-feature fusion vector | KernelSHAP (model-agnostic, sampled) |

Inputs the selected model provably does not consume are reported in
`unattributed_inputs` with the reason, never given an invented score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from explainability.shap_explainer.config import ShapConfig

logger = logging.getLogger(__name__)


class ExplainerNotAvailableError(RuntimeError):
    """Raised when the artifacts SHAP needs are missing."""


class InvalidExplanationInputError(ValueError):
    """Raised when the row handed to the explainer is malformed."""


@dataclass
class FeatureContribution:
    """One feature's signed effect on the fused risk score."""

    feature: str
    label: str
    value: float | None
    #: Signed SHAP value, already expressed in fused-risk probability units.
    contribution: float
    #: Share of the total absolute attribution, signed, in [-1, 1].
    share: float
    impact: str
    direction: str
    branch: str
    #: True when the model was scored on a training median because the
    #: record carried no observation. The attribution is real, but it
    #: describes the median, so the narrative layer skips these.
    imputed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": None if self.value is None else round(float(self.value), 4),
            "contribution": round(float(self.contribution), 6),
            "contribution_pct": round(float(self.share) * 100, 1),
            "impact": self.impact,
            "direction": self.direction,
            "branch": self.branch,
            "imputed": self.imputed,
        }


@dataclass
class ShapExplanation:
    """Everything the narrative and report layers need from SHAP."""

    contributions: list[FeatureContribution]
    base_value: float
    #: Risk reconstructed by the explained models: base + sum(attributions).
    predicted_risk: float
    method: str
    branch_weights: dict[str, float]
    branch_contributions: dict[str, float]
    #: What the stored fusion strategy returns for this row. Differs from
    #: `predicted_risk` on dataset rows, which carry out-of-fold upstream
    #: probabilities rather than the shipped artifacts' output.
    served_risk: float | None = None
    unattributed_inputs: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def top(self, count: int) -> list[FeatureContribution]:
        return sorted(self.contributions, key=lambda c: -abs(c.contribution))[:count]

    def to_dict(self, top_k: int) -> dict[str, Any]:
        return {
            "method": self.method,
            "base_value": round(float(self.base_value), 6),
            "predicted_risk": round(float(self.predicted_risk), 6),
            "served_risk": None if self.served_risk is None else round(float(self.served_risk), 6),
            "branch_weights": {k: round(float(v), 4) for k, v in self.branch_weights.items()},
            "branch_contributions": {
                k: round(float(v), 6) for k, v in self.branch_contributions.items()
            },
            "top_features": [c.to_dict() for c in self.top(top_k)],
            "all_features": [c.to_dict() for c in self.contributions],
            "unattributed_inputs": self.unattributed_inputs,
            "notes": self.notes,
        }


class FusionShapExplainer:
    """Computes SHAP attributions for a trained Phase 5 fusion bundle."""

    def __init__(self, bundle: dict[str, Any], config: ShapConfig | None = None) -> None:
        self.config = config or ShapConfig()
        self.bundle = bundle
        self.strategy = bundle["strategy"]
        self._weather_bundle: dict[str, Any] | None = None
        self._tree_explainer: Any = None
        self._kernel_explainer: Any = None
        self._region_coordinates_cache: dict[str, dict[str, float]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def explain(self, row: pd.DataFrame) -> ShapExplanation:
        """Attribute one region-day's fused risk to its inputs.

        Args:
            row: single-row frame carrying the fusion feature columns and
                the upstream probability columns.
        """
        if not isinstance(row, pd.DataFrame) or len(row) != 1:
            raise InvalidExplanationInputError(
                f"Expected a single-row DataFrame, got {type(row).__name__} of length "
                f"{len(row) if hasattr(row, '__len__') else 'n/a'}"
            )

        from ai_models.fusion_model.fusion import FeatureFusion, NeuralFusion, WeightedFusion

        if isinstance(self.strategy, WeightedFusion):
            return self._explain_weighted(row)
        if isinstance(self.strategy, FeatureFusion):
            return self._explain_tree_fusion(row)
        if isinstance(self.strategy, NeuralFusion):
            return self._explain_kernel_fusion(row)
        raise ExplainerNotAvailableError(
            f"No SHAP path implemented for fusion strategy {type(self.strategy).__name__}"
        )

    # ------------------------------------------------------------------
    # Case A — weighted late fusion: attribute through the branches
    # ------------------------------------------------------------------
    def _explain_weighted(self, row: pd.DataFrame) -> ShapExplanation:
        weight = float(self.strategy.weather_weight)
        weights = {"weather": weight, "satellite": 1.0 - weight}

        weather_shap, weather_base, weather_risk, scored, imputed = self._weather_branch_shap(row)
        contributions = [
            self._contribution(
                feature=name,
                # The value the forest actually saw, median-filled or not.
                value=float(scored.iloc[0][name]),
                contribution=float(value) * weight,
                branch="weather",
                imputed=name in imputed,
            )
            for name, value in weather_shap.items()
        ]

        # The satellite branch has no per-feature tabular attribution — it
        # reads pixels. Its effect on the fused score is exact and analytic:
        # its weight times how far its risk sits from the branch baseline.
        satellite_risk = self._risk_from(row, "satellite_prob")
        satellite_base = self._satellite_baseline()
        satellite_contribution = (1.0 - weight) * (satellite_risk - satellite_base)
        if abs(1.0 - weight) > 1e-9:
            contributions.append(
                self._contribution(
                    feature="satellite_risk_score",
                    value=satellite_risk,
                    contribution=satellite_contribution,
                    branch="satellite",
                )
            )

        contributions = self._rescore_shares(contributions)
        base_value = weight * weather_base + (1.0 - weight) * satellite_base
        # Computed from the branches that were actually explained, so that
        # base + sum(attributions) reconciles exactly. The stored column may
        # differ: dataset rows carry out-of-fold probabilities, while serving
        # (and this explanation) use the shipped Phase 3 artifact.
        predicted = weight * weather_risk + (1.0 - weight) * satellite_risk
        served = float(self.strategy.predict_proba(row)[0][1:].sum())

        notes = [
            "Attribution follows the fusion chain rule: SHAP on the branch that reads each "
            f"input, scaled by that branch's weight (weather {weight:.2f}, "
            f"satellite {1 - weight:.2f}).",
            "Weather attributions are exact TreeSHAP values on the Phase 3 forest, so the "
            "baseline plus the attributions reconstruct the explained risk exactly.",
        ]
        if abs(served - predicted) > 1e-6:
            notes.append(
                f"The stored fusion output for this row is {served:.4f} against the explained "
                f"{predicted:.4f}. Dataset rows carry out-of-fold upstream probabilities, while "
                "this explanation runs the shipped Phase 3 artifact — the one that serves live "
                "predictions. The gap is the out-of-fold correction, not an explanation error."
            )
        unattributed: list[dict[str, str]] = []
        if imputed:
            unattributed.append(
                {
                    "inputs": ", ".join(imputed),
                    "reason": "not present in the region-day record; the forest was scored on "
                              "the training median, so any attribution shown reflects that "
                              "median rather than an observed value",
                }
            )
        if abs(1.0 - weight) < 1e-9:
            notes.append(
                "The validation weight sweep gave the satellite branch weight 0.00, so it "
                "contributes exactly nothing to this decision. The Grad-CAM heatmap therefore "
                "explains the satellite model's own same-day reading, not the fused forecast."
            )
            unattributed.append(
                {
                    "inputs": "satellite scene features and satellite risk score",
                    "reason": "satellite branch weight is 0.00 in the selected fusion",
                }
            )
        # Scene statistics only enter a feature-level fusion, never this one.
        scene_features = [
            name
            for name in ("cloud_density", "brightness_mean", "spatial_dispersion",
                         "cold_top_fraction", "cloud_growth_rate")
            if name in row.columns
        ]
        if scene_features:
            unattributed.append(
                {
                    "inputs": ", ".join(scene_features),
                    "reason": "the selected weighted fusion consumes only the two upstream "
                              "probability vectors, so these columns cannot move its output",
                }
            )

        return ShapExplanation(
            contributions=contributions,
            base_value=base_value,
            predicted_risk=predicted,
            served_risk=served,
            method="chain-rule TreeSHAP through the weather branch",
            branch_weights=weights,
            branch_contributions={
                "weather": weight * (weather_risk - weather_base),
                "satellite": satellite_contribution,
            },
            unattributed_inputs=unattributed,
            notes=notes,
        )

    def _weather_branch_shap(
        self, row: pd.DataFrame
    ) -> tuple[dict[str, float], float, float, pd.DataFrame, list[str]]:
        """Exact TreeSHAP over the Phase 3 forest, reduced to risk units.

        Returns the attributions, the branch baseline and prediction, the
        exact frame the forest was scored on, and the names that had to be
        filled from training medians — the caller reports those rather than
        showing a blank value beside a real attribution.
        """
        import shap

        bundle = self._load_weather_bundle()
        names = list(bundle["feature_names"])
        medians = bundle.get("feature_medians", {})

        # The fusion dataset aggregates to region-days and drops the point
        # coordinates the Phase 3 forest was trained on. Falling straight
        # back to the global median put ~20% of the attribution mass on a
        # constant, so resolve the region's real centroid first.
        coordinates = self._region_coordinates(_region_of(row))

        values: dict[str, float] = {}
        imputed: list[str] = []
        for name in names:
            supplied = self._value_of(row, name)
            if supplied is None:
                supplied = coordinates.get(name)
            if supplied is None:
                supplied = float(medians.get(name, 0.0))
                imputed.append(name)
            values[name] = float(supplied)
        frame = pd.DataFrame([values])

        if self._tree_explainer is None:
            self._tree_explainer = shap.TreeExplainer(bundle["estimator"])

        values = np.asarray(
            self._tree_explainer.shap_values(frame, check_additivity=False), dtype=np.float64
        )
        # TreeSHAP returns (samples, features, classes) for multiclass forests.
        if values.ndim == 3:
            risk_values = values[0, :, 1:].sum(axis=1)
        else:  # binary estimator — the positive column is already the risk
            risk_values = values[0]

        expected = np.atleast_1d(np.asarray(self._tree_explainer.expected_value, dtype=np.float64))
        base = float(expected[1:].sum()) if expected.size >= 3 else float(expected[-1])

        probabilities = bundle["estimator"].predict_proba(frame)[0]
        risk = float(np.asarray(probabilities)[1:].sum())
        return dict(zip(names, risk_values)), base, risk, frame, imputed

    def _satellite_baseline(self) -> float:
        """Mean satellite risk over the training record — the branch's prior."""
        frame = self._reference_frame()
        if frame is None or "satellite_risk_score" not in frame:
            return 0.0
        return float(frame["satellite_risk_score"].mean())

    # ------------------------------------------------------------------
    # Case B — feature-level fusion: SHAP directly on the fusion estimator
    # ------------------------------------------------------------------
    def _explain_tree_fusion(self, row: pd.DataFrame) -> ShapExplanation:
        import shap

        names = list(self.bundle["feature_names"])
        frame = row[names].astype(float)

        if self._tree_explainer is None:
            self._tree_explainer = shap.TreeExplainer(self.strategy.estimator)
        values = np.asarray(
            self._tree_explainer.shap_values(frame, check_additivity=False), dtype=np.float64
        )
        risk_values = values[0, :, 1:].sum(axis=1) if values.ndim == 3 else values[0]

        expected = np.atleast_1d(np.asarray(self._tree_explainer.expected_value, dtype=np.float64))
        base = float(expected[1:].sum()) if expected.size >= 3 else float(expected[-1])

        contributions = self._rescore_shares(
            [
                self._contribution(
                    feature=name,
                    value=self._value_of(row, name),
                    contribution=float(value),
                    branch=self._branch_of(name),
                )
                for name, value in zip(names, risk_values)
            ]
        )
        return ShapExplanation(
            contributions=contributions,
            base_value=base,
            predicted_risk=float(self.strategy.predict_proba(row)[0][1:].sum()),
            method="exact TreeSHAP on the feature-level fusion",
            branch_weights={"weather": float("nan"), "satellite": float("nan")},
            branch_contributions=self._branch_totals(contributions),
            notes=[
                "The feature-level fusion reads every input directly, so attributions are "
                "exact TreeSHAP values on the fusion estimator itself.",
            ],
        )

    # ------------------------------------------------------------------
    # Case C — neural fusion: model-agnostic KernelSHAP
    # ------------------------------------------------------------------
    def _explain_kernel_fusion(self, row: pd.DataFrame) -> ShapExplanation:
        import shap

        names = list(self.bundle["feature_names"])
        background = self._background_sample(names)
        frame = row[names].astype(float)

        def predict(matrix: np.ndarray) -> np.ndarray:
            batch = pd.DataFrame(matrix, columns=names)
            return self.strategy.predict_proba(batch)[:, 1:].sum(axis=1)

        if self._kernel_explainer is None:
            self._kernel_explainer = shap.KernelExplainer(predict, background)
        values = np.asarray(
            self._kernel_explainer.shap_values(frame.to_numpy(), silent=True), dtype=np.float64
        )
        risk_values = values[0] if values.ndim > 1 else values

        contributions = self._rescore_shares(
            [
                self._contribution(
                    feature=name,
                    value=self._value_of(row, name),
                    contribution=float(value),
                    branch=self._branch_of(name),
                )
                for name, value in zip(names, risk_values)
            ]
        )
        return ShapExplanation(
            contributions=contributions,
            base_value=float(np.asarray(self._kernel_explainer.expected_value).ravel()[0]),
            predicted_risk=float(self.strategy.predict_proba(row)[0][1:].sum()),
            method=f"KernelSHAP on the neural fusion ({len(background)} background rows)",
            branch_weights={"weather": float("nan"), "satellite": float("nan")},
            branch_contributions=self._branch_totals(contributions),
            notes=[
                "KernelSHAP is sampled, not exact: values are approximations whose precision "
                f"is bounded by the {len(background)}-row background set.",
            ],
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _contribution(
        self, *, feature: str, value: float | None, contribution: float, branch: str,
        imputed: bool = False,
    ) -> FeatureContribution:
        inverted = feature in self.config.inverted_features
        rising = contribution > 0
        if abs(contribution) < 1e-12:
            direction = "no effect"
        elif inverted:
            direction = "falling — raises risk" if rising else "rising — lowers risk"
        else:
            direction = "raises risk" if rising else "lowers risk"
        return FeatureContribution(
            feature=feature,
            label=self.config.label_for(feature),
            value=value,
            contribution=contribution,
            share=0.0,  # filled by _rescore_shares once the total is known
            impact="Low",
            direction=direction,
            branch=branch,
            imputed=imputed,
        )

    def _rescore_shares(self, contributions: list[FeatureContribution]) -> list[FeatureContribution]:
        """Express each attribution as a signed share of the total magnitude,
        then label its impact from that measured share."""
        total = sum(abs(c.contribution) for c in contributions)
        for item in contributions:
            item.share = 0.0 if total <= 0 else item.contribution / total
            item.impact = self.config.impact_for(item.share)
        return contributions

    @staticmethod
    def _branch_totals(contributions: list[FeatureContribution]) -> dict[str, float]:
        totals: dict[str, float] = {}
        for item in contributions:
            totals[item.branch] = totals.get(item.branch, 0.0) + item.contribution
        return totals

    @staticmethod
    def _branch_of(feature: str) -> str:
        satellite = {
            "cloud_density", "brightness_mean", "brightness_std", "spatial_dispersion",
            "cold_top_fraction", "cloud_growth_rate", "valid_fraction", "satellite_risk_score",
        }
        return "satellite" if feature in satellite else "weather"

    @staticmethod
    def _value_of(row: pd.DataFrame, name: str, default: float | None = None) -> float | None:
        if name not in row.columns:
            return default
        value = row.iloc[0][name]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)

    @staticmethod
    def _risk_from(row: pd.DataFrame, prefix: str) -> float:
        columns = [c for c in row.columns if c.startswith(prefix)]
        if not columns:
            return 0.0
        ordered = sorted(columns, key=lambda c: int(c.rsplit("_", 1)[-1]))
        return float(sum(float(row.iloc[0][c]) for c in ordered[1:]))

    def _load_weather_bundle(self) -> dict[str, Any]:
        if self._weather_bundle is None:
            from ai_models.baseline.config import BaselineConfig
            from ai_models.baseline.model import BUNDLE_FILENAME_TEMPLATE, load_bundle

            path = self.config.saved_models_dir / BUNDLE_FILENAME_TEMPLATE.format(
                version=BaselineConfig().model_version
            )
            if not path.exists():
                raise ExplainerNotAvailableError(
                    f"Phase 3 model missing at {path}; SHAP cannot attribute the weather branch."
                )
            self._weather_bundle = load_bundle(path)
        return self._weather_bundle

    def _region_coordinates(self, region: str | None) -> dict[str, float]:
        """Mean latitude/longitude of a region's observation points.

        The Phase 3 forest reads coordinates as features, so a region-day
        must be scored at the region's real location rather than at the
        dataset-wide median, which belongs to no region at all.
        """
        if region is None:
            return {}
        if self._region_coordinates_cache is None:
            self._region_coordinates_cache = {}
            from ai_models.fusion_model.config import FusionConfig

            path = FusionConfig().weather_dataset_path
            if not path.exists():
                logger.warning("Weather dataset absent at %s; coordinates unresolved", path)
                return {}
            points = pd.read_parquet(path, columns=["region", "latitude", "longitude"])
            means = points.groupby("region")[["latitude", "longitude"]].mean()
            self._region_coordinates_cache = {
                str(name): {"latitude": float(values["latitude"]),
                            "longitude": float(values["longitude"])}
                for name, values in means.iterrows()
            }
        return self._region_coordinates_cache.get(str(region), {})

    def _reference_frame(self) -> pd.DataFrame | None:
        if not self.config.fusion_dataset_path.exists():
            logger.warning("Fusion dataset absent at %s", self.config.fusion_dataset_path)
            return None
        return pd.read_parquet(self.config.fusion_dataset_path)

    def _background_sample(self, names: list[str]) -> np.ndarray:
        frame = self._reference_frame()
        if frame is None:
            raise ExplainerNotAvailableError(
                "KernelSHAP needs the fusion dataset as a background distribution."
            )
        sample = frame[names].astype(float)
        if len(sample) > self.config.background_size:
            sample = sample.sample(self.config.background_size, random_state=0)
        return sample.to_numpy()

def _region_of(row: pd.DataFrame) -> str | None:
    """The region name carried by a fusion row, if any."""
    if "region" not in row.columns:
        return None
    value = row.iloc[0]["region"]
    return None if value is None else str(value)


def load_fusion_bundle(config: ShapConfig | None = None) -> dict[str, Any]:
    """Load the trained Phase 5 bundle, or explain why it is unavailable."""
    import joblib

    from ai_models.fusion_model.config import FUSION_BUNDLE_TEMPLATE, FusionConfig

    config = config or ShapConfig()
    fusion_config = FusionConfig()
    path: Path = config.saved_models_dir / FUSION_BUNDLE_TEMPLATE.format(
        version=fusion_config.model_version
    )
    if not path.exists():
        raise ExplainerNotAvailableError(
            f"No fusion model at {path}. Run `python -m ai_models.fusion_model.train` first."
        )
    return joblib.load(path)

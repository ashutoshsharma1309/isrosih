"""
Explainability layer tests (Phase 6).

Hermetic by construction: every fixture builds its own miniature Phase 3
bundle, Phase 4 checkpoint and Phase 5 fusion bundle in `tmp_path`, so the
suite exercises the real SHAP, Captum and narrative code without depending
on the shipped artifacts.

The load-bearing tests are the honesty guards:

- `test_weighted_fusion_attribution_is_additive` — attributions must
  reconstruct the explained risk, not merely look plausible.
- `test_weighted_fusion_reports_unconsumed_inputs` — inputs the model
  cannot read are declared, never given an invented score.
- `test_narrative_omits_imputed_drivers` — prose never asserts something
  about a value that was median-filled rather than observed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from ai_models.fusion_model.config import FusionConfig
from ai_models.fusion_model.fusion import FeatureFusion, NeuralFusion, WeightedFusion
from ai_models.fusion_model.historical_match import HistoricalMatcher
from explainability.explanation_generator import templates
from explainability.explanation_generator.generator import ExplanationGenerator
from explainability.gradcam_explainer.config import GradCamConfig
from explainability.gradcam_explainer.gradcam import (
    GradCamNotAvailableError,
    InvalidSceneError,
    SatelliteGradCam,
)
from explainability.historical_explainer import (
    HistoricalExplainer,
    HistoricalExplanationUnavailableError,
)
from explainability.shap_explainer.config import ShapConfig
from explainability.shap_explainer.explainer import (
    ExplainerNotAvailableError,
    FusionShapExplainer,
    InvalidExplanationInputError,
)

FUSION = FusionConfig()


# ---------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------
def _fusion_frame(rows: int = 120) -> pd.DataFrame:
    """Region-days where risk genuinely rises with humidity and rainfall."""
    rng = np.random.default_rng(11)
    records = []
    for i in range(rows):
        wet = i % 3 == 0
        humidity = float(np.clip(rng.normal(93 if wet else 60, 3), 0, 100))
        rain = 70.0 if wet else 2.0
        weather_risk = float(np.clip(rng.normal(0.85 if wet else 0.15, 0.05), 0, 1))
        satellite_risk = float(np.clip(rng.normal(0.7 if wet else 0.3, 0.08), 0, 1))
        cloud = float(np.clip(rng.normal(0.9 if wet else 0.3, 0.05), 0, 1))
        records.append(
            {
                "region": ["Kerala", "Mumbai"][i % 2],
                "date": f"20{10 + i // 60:02d}-{1 + (i % 12):02d}-{1 + (i % 28):02d}",
                "temperature_c": float(rng.normal(28, 1.5)),
                "humidity_pct": humidity,
                "pressure_hpa": float(rng.normal(1000 if wet else 1010, 2)),
                "wind_speed_ms": float(rng.normal(7 if wet else 3, 1)),
                "wind_direction_deg": float(rng.uniform(0, 360)),
                "cloud_cover_pct": cloud * 100,
                "rain_sum_1d": rain,
                "rain_sum_3d": rain * 2,
                "rain_sum_7d": rain * 3,
                "rain_sum_30d": rain * 6,
                "rain_trend_3d": 20.0 if wet else -2.0,
                "season_sin": float(np.sin(i)),
                "season_cos": float(np.cos(i)),
                "latitude": 10.0 if i % 2 == 0 else 19.0,
                "longitude": 76.3 if i % 2 == 0 else 72.8,
                "cloud_density": cloud,
                "brightness_mean": cloud * 0.9,
                "brightness_std": 0.15,
                "spatial_dispersion": 0.85,
                "cold_top_fraction": cloud * 0.4,
                "cloud_growth_rate": 0.5,
                "valid_fraction": 1.0,
                "rainfall_mm": rain,
                "weather_risk_score": weather_risk,
                "satellite_risk_score": satellite_risk,
                "weather_prob_0": 1 - weather_risk,
                "weather_prob_1": weather_risk,
                "weather_prob_2": 0.0,
                "satellite_prob_0": 1 - satellite_risk,
                "satellite_prob_1": satellite_risk,
                "satellite_prob_2": 0.0,
                FUSION.target_column: 1 if wet else 0,
            }
        )
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return _fusion_frame()


@pytest.fixture()
def row(frame: pd.DataFrame) -> pd.DataFrame:
    """A wet region-day — the interesting case to explain."""
    return frame[frame[FUSION.target_column] == 1].iloc[[0]]


@pytest.fixture()
def shap_config(tmp_path: Path, frame: pd.DataFrame) -> ShapConfig:
    """Config pointing at a self-contained artifact directory."""
    from ai_models.baseline.model import BUNDLE_FILENAME_TEMPLATE
    from sklearn.ensemble import RandomForestClassifier

    models = tmp_path / "saved_models"
    models.mkdir(parents=True, exist_ok=True)

    weather_features = [
        "temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_ms",
        "wind_direction_deg", "cloud_cover_pct", "latitude", "longitude",
        "rain_sum_1d", "rain_sum_3d", "rain_sum_7d", "rain_sum_30d",
        "rain_trend_3d", "season_sin", "season_cos",
    ]
    estimator = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=0)
    estimator.fit(frame[weather_features], frame[FUSION.target_column])
    joblib.dump(
        {
            "version": "v1",
            "model_name": "random_forest",
            "estimator": estimator,
            "feature_names": weather_features,
            "feature_medians": frame[weather_features].median().to_dict(),
            "label_names": FUSION.label_names,
        },
        models / BUNDLE_FILENAME_TEMPLATE.format(version="v1"),
    )

    dataset = tmp_path / "fusion_dataset.parquet"
    frame.to_parquet(dataset, index=False)
    return replace(
        ShapConfig(),
        saved_models_dir=models,
        fusion_dataset_path=dataset,
        reports_dir=tmp_path / "reports_shap",
    )


def _bundle(strategy, frame: pd.DataFrame) -> dict:
    target = frame[FUSION.target_column]
    return {
        "version": "test",
        "model_name": getattr(strategy, "name", "strategy"),
        "approach": strategy.approach,
        "strategy": strategy.fit(frame, target, FUSION),
        "matcher": HistoricalMatcher().fit(frame, FUSION),
        "feature_names": FUSION.feature_columns(),
        "match_features": list(HistoricalMatcher().features),
        "label_names": FUSION.label_names,
        "event_names": FUSION.event_names,
        "feature_medians": frame[FUSION.feature_columns()].median().to_dict(),
    }


@pytest.fixture()
def weighted_bundle(frame: pd.DataFrame) -> dict:
    return _bundle(WeightedFusion(1.0), frame)


# ---------------------------------------------------------------------
# SHAP — weighted late fusion (the shipped shape)
# ---------------------------------------------------------------------
def test_weighted_fusion_attribution_is_additive(weighted_bundle, shap_config, row):
    """base + sum(attributions) must reconstruct the explained risk exactly."""
    explanation = FusionShapExplainer(weighted_bundle, shap_config).explain(row)
    total = sum(c.contribution for c in explanation.contributions)
    assert explanation.base_value + total == pytest.approx(explanation.predicted_risk, abs=1e-6)


def test_weighted_fusion_scales_by_branch_weight(frame, shap_config, row):
    """Halving the weather weight must halve every weather attribution."""
    full = FusionShapExplainer(_bundle(WeightedFusion(1.0), frame), shap_config).explain(row)
    half = FusionShapExplainer(_bundle(WeightedFusion(0.5), frame), shap_config).explain(row)

    full_by_name = {c.feature: c.contribution for c in full.contributions if c.branch == "weather"}
    half_by_name = {c.feature: c.contribution for c in half.contributions if c.branch == "weather"}
    assert full_by_name and half_by_name
    for name, value in full_by_name.items():
        assert half_by_name[name] == pytest.approx(value * 0.5, abs=1e-9)


def test_weighted_fusion_reports_unconsumed_inputs(weighted_bundle, shap_config, row):
    """Inputs the blender cannot read are declared, not scored."""
    explanation = FusionShapExplainer(weighted_bundle, shap_config).explain(row)

    attributed = {c.feature for c in explanation.contributions}
    assert "cloud_density" not in attributed, "a w=1.0 blend cannot read scene statistics"

    reasons = " ".join(entry["reason"] for entry in explanation.unattributed_inputs)
    assert "cloud_density" in " ".join(e["inputs"] for e in explanation.unattributed_inputs)
    assert reasons, "unconsumed inputs must carry an explanation"
    assert explanation.branch_weights["satellite"] == pytest.approx(0.0)
    assert explanation.branch_contributions["satellite"] == pytest.approx(0.0)


def test_shares_and_impacts_are_derived_from_measurement(weighted_bundle, shap_config, row):
    explanation = FusionShapExplainer(weighted_bundle, shap_config).explain(row)
    total = sum(abs(c.contribution) for c in explanation.contributions)
    for item in explanation.contributions:
        assert item.share == pytest.approx(item.contribution / total, abs=1e-9)
        assert item.impact in {"Critical", "High", "Medium", "Low"}
    assert sum(abs(c.share) for c in explanation.contributions) == pytest.approx(1.0, abs=1e-6)


def test_region_coordinates_are_resolved_not_imputed(weighted_bundle, shap_config, row):
    """Region-days must be scored at their real location."""
    explanation = FusionShapExplainer(weighted_bundle, shap_config).explain(row)
    coordinates = {c.feature: c for c in explanation.contributions if c.feature in {"latitude", "longitude"}}
    assert coordinates, "the weather model reads coordinates"
    for item in coordinates.values():
        assert not item.imputed


# ---------------------------------------------------------------------
# SHAP — the other two fusion shapes
# ---------------------------------------------------------------------
def test_feature_fusion_attributes_scene_statistics(frame, shap_config, row):
    """A feature-level fusion does read scene statistics, so they get scores."""
    bundle = _bundle(FeatureFusion("random_forest", 0), frame)
    explanation = FusionShapExplainer(bundle, shap_config).explain(row)

    attributed = {c.feature for c in explanation.contributions}
    assert "cloud_density" in attributed
    assert "humidity_pct" in attributed
    assert explanation.method.startswith("exact TreeSHAP")


def test_neural_fusion_uses_kernel_shap(frame, shap_config, row):
    bundle = _bundle(NeuralFusion(0), frame)
    explanation = FusionShapExplainer(bundle, shap_config).explain(row)
    assert "KernelSHAP" in explanation.method
    assert len(explanation.contributions) == len(FUSION.feature_columns())
    assert any(abs(c.contribution) > 0 for c in explanation.contributions)


def test_unknown_strategy_is_rejected(weighted_bundle, shap_config, row):
    weighted_bundle["strategy"] = object()
    with pytest.raises(ExplainerNotAvailableError):
        FusionShapExplainer(weighted_bundle, shap_config).explain(row)


def test_missing_weather_model_is_reported(weighted_bundle, shap_config, row, tmp_path):
    empty = replace(shap_config, saved_models_dir=tmp_path / "absent")
    with pytest.raises(ExplainerNotAvailableError):
        FusionShapExplainer(weighted_bundle, empty).explain(row)


@pytest.mark.parametrize("bad", [pd.DataFrame(), "not-a-frame", None])
def test_invalid_explanation_input_rejected(weighted_bundle, shap_config, bad):
    with pytest.raises(InvalidExplanationInputError):
        FusionShapExplainer(weighted_bundle, shap_config).explain(bad)


# ---------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------
def _write_scene(path: Path, cloudy: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(20, 70, size=(256, 256, 3), dtype=np.uint8)
    columns = int(256 * cloudy)
    image[:, :columns] = rng.integers(190, 255, size=(256, columns, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


@pytest.fixture()
def gradcam_config(tmp_path: Path) -> GradCamConfig:
    """A real (untrained) Phase 4 checkpoint, so Captum has something to read."""
    from ai_models.satellite_model.config import SatelliteConfig
    from ai_models.satellite_model.model import CustomCNN, save_checkpoint

    models = tmp_path / "saved_models"
    save_checkpoint(
        config=replace(SatelliteConfig(), saved_models_dir=models),
        model_name="custom_cnn",
        model=CustomCNN(num_classes=3),
        metrics={},
        dataset_info={},
        gradcam_target_layer="features.3",
    )
    return replace(
        GradCamConfig(), saved_models_dir=models, reports_dir=tmp_path / "reports_gradcam"
    )


def test_gradcam_produces_a_real_normalised_map(gradcam_config, tmp_path):
    scene = tmp_path / "scene.jpg"
    _write_scene(scene, cloudy=0.8)

    explanation = SatelliteGradCam(gradcam_config).explain(scene)
    assert explanation.heatmap.shape == (256, 256)
    assert 0.0 <= float(explanation.heatmap.min())
    assert float(explanation.heatmap.max()) <= 1.0
    assert explanation.target_layer == "features.3"
    assert 0.0 <= explanation.satellite_risk <= 1.0
    assert sum(explanation.class_probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_gradcam_is_deterministic_for_a_scene(gradcam_config, tmp_path):
    """Attribution must come from gradients, not randomness."""
    scene = tmp_path / "scene.jpg"
    _write_scene(scene, cloudy=0.7)
    cam = SatelliteGradCam(gradcam_config)
    assert np.allclose(cam.explain(scene).heatmap, cam.explain(scene).heatmap)


def test_gradcam_regions_are_named_and_bounded(gradcam_config, tmp_path):
    scene = tmp_path / "scene.jpg"
    _write_scene(scene, cloudy=0.6)
    explanation = SatelliteGradCam(gradcam_config).explain(scene)
    for index, region in enumerate(explanation.regions, start=1):
        assert region.name == f"cloud_cluster_area_{index}"
        x0, y0, x1, y1 = region.bbox
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
        assert region.area_share >= gradcam_config.min_region_area
    assert len(explanation.regions) <= gradcam_config.max_regions


def test_gradcam_rejects_unreadable_scene(gradcam_config, tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    with pytest.raises(InvalidSceneError):
        SatelliteGradCam(gradcam_config).explain(broken)


def test_gradcam_rejects_out_of_range_class(gradcam_config, tmp_path):
    scene = tmp_path / "scene.jpg"
    _write_scene(scene, cloudy=0.5)
    with pytest.raises(InvalidSceneError):
        SatelliteGradCam(gradcam_config).explain(scene, target_class=9)


def test_missing_checkpoint_is_reported(tmp_path):
    config = replace(GradCamConfig(), saved_models_dir=tmp_path / "absent")
    scene = tmp_path / "scene.jpg"
    _write_scene(scene, cloudy=0.5)
    with pytest.raises(GradCamNotAvailableError):
        SatelliteGradCam(config).explain(scene)


def test_overlay_is_written(gradcam_config, tmp_path):
    from explainability.gradcam_explainer.visualization import save_overlay

    scene = tmp_path / "scene.jpg"
    _write_scene(scene, cloudy=0.75)
    explanation = SatelliteGradCam(gradcam_config).explain(scene)
    path = save_overlay(scene, explanation, gradcam_config)
    assert Path(path).exists() and Path(path).stat().st_size > 0


# ---------------------------------------------------------------------
# Historical explanation
# ---------------------------------------------------------------------
def test_historical_explainer_returns_ranked_analogues(weighted_bundle, row):
    explanation = HistoricalExplainer(weighted_bundle).explain(row, FUSION, top_k=3)
    assert explanation.matches
    assert explanation.reference_size > 0
    similarities = [m["similarity"] for m in explanation.matches]
    assert similarities == sorted(similarities, reverse=True)


def test_historical_explainer_requires_a_matcher():
    with pytest.raises(HistoricalExplanationUnavailableError):
        HistoricalExplainer({"matcher": None})


def test_historical_explainer_rejects_missing_conditions(weighted_bundle):
    with pytest.raises(HistoricalExplanationUnavailableError):
        HistoricalExplainer(weighted_bundle).explain(
            pd.DataFrame([{"humidity_pct": 90.0}]), FUSION
        )


# ---------------------------------------------------------------------
# Narrative and unified payload
# ---------------------------------------------------------------------
def test_unified_payload_has_the_phase6_contract(weighted_bundle, shap_config, gradcam_config, row, tmp_path):
    scene = tmp_path / "scene.jpg"
    _write_scene(scene, cloudy=0.85)
    generator = ExplanationGenerator(weighted_bundle, shap_config, gradcam_config)
    payload = generator.explain(row, image_path=scene).to_dict()

    for key in (
        "prediction", "probability", "confidence", "risk_level",
        "top_features", "satellite_regions", "explanation",
    ):
        assert key in payload, f"missing contract field {key}"
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["confidence"] in {"Low", "Medium", "High"}
    assert payload["risk_level"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert all(isinstance(name, str) for name in payload["top_features"])
    assert payload["explanation"].strip()


def test_narrative_omits_imputed_drivers(weighted_bundle, shap_config, row):
    """Prose must not assert anything about a median-filled value."""
    stripped = row.drop(columns=["humidity_pct"])
    generator = ExplanationGenerator(weighted_bundle, shap_config)
    explanation = generator.explain(stripped)

    imputed = [c for c in explanation.shap.contributions if c.imputed]
    assert imputed, "dropping a column should force an imputation"
    for item in imputed:
        assert item.label.lower() not in explanation.narrative.lower()


def test_narrative_declares_a_zero_weight_branch(weighted_bundle, shap_config, gradcam_config, row, tmp_path):
    """A branch with no weight must be reported as not having influenced the call."""
    scene = tmp_path / "scene.jpg"
    _write_scene(scene, cloudy=0.9)
    generator = ExplanationGenerator(weighted_bundle, shap_config, gradcam_config)
    explanation = generator.explain(row, image_path=scene)
    assert "zero weight" in explanation.narrative


def test_confidence_decomposes_into_measured_factors(weighted_bundle, shap_config, row):
    generator = ExplanationGenerator(weighted_bundle, shap_config)
    confidence = generator.explain(row).confidence.to_dict()
    assert 0.0 <= confidence["confidence_pct"] <= 100.0
    assert confidence["factors"]["data_quality"] in {"Good", "Fair", "Poor"}
    assert 0.0 <= confidence["factors"]["data_quality_score"] <= 1.0


def test_missing_scene_degrades_without_failing(weighted_bundle, shap_config, gradcam_config, row):
    """A missing image must not sink the whole explanation."""
    generator = ExplanationGenerator(weighted_bundle, shap_config, gradcam_config)
    explanation = generator.explain(row, image_path="/nonexistent/scene.jpg")
    assert explanation.gradcam is None
    assert explanation.narrative.strip()
    assert any("satellite explanation" in c for c in explanation.caveats)


# ---------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------
def test_pressure_direction_is_inverted():
    """Falling pressure raises risk — the phrasing must reflect that."""
    assert templates.describe_driver("pressure_hpa", "Atmospheric pressure", True) == (
        "falling atmospheric pressure"
    )


def test_join_clauses_reads_as_prose():
    assert templates.join_clauses(["a", "b", "c"]) == "a, b and c"
    assert templates.join_clauses(["a"]) == "a"
    assert templates.join_clauses([]) == ""


def test_drivers_sentence_is_well_formed():
    sentence = templates.drivers_sentence(["heavy rain"], ["dry month"])
    assert sentence.startswith("Risk was raised")
    assert "; ," not in sentence and ", while" in sentence
    assert sentence.endswith(".")


def test_satellite_sentence_declares_zero_weight():
    sentence = templates.satellite_sentence([], "localised", weight=0.0, risk=0.84)
    assert "zero weight" in sentence and "84%" in sentence

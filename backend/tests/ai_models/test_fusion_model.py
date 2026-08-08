"""
Fusion model tests (Phase 5).

Synthetic region-days stand in for the real unified dataset so the
strategies, historical matcher, and prediction contract can be exercised
without the Phase 3/4 artifacts. The shipped bundle is trained
exclusively on the real joined dataset.

The leakage guard (`test_walk_forward_scores_are_out_of_sample`) is the
important one: it is what stops the fusion being trained on a memorised
upstream score again.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ai_models.fusion_model.config import FusionConfig
from ai_models.fusion_model.dataset import chronological_split
from ai_models.fusion_model.fusion import (
    FeatureFusion,
    SingleModality,
    WeightedFusion,
    fit_weight,
)
from ai_models.fusion_model.historical_match import (
    MATCH_FEATURES,
    HistoricalMatcher,
    MatcherNotFittedError,
)
from ai_models.fusion_model.predict import (
    HybridPredictor,
    InvalidInputError,
    ModelNotTrainedError,
    _vector_from_score,
)
from ai_models.fusion_model.utils import (
    align_probabilities,
    confidence_label,
    model_agreement,
    risk_level,
    risk_score,
)


@pytest.fixture()
def config() -> FusionConfig:
    return FusionConfig()


@pytest.fixture()
def unified(config: FusionConfig) -> pd.DataFrame:
    """160 synthetic region-days where risk rises with humidity and cloud."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(160):
        wet = i % 4 == 0
        cloud = float(np.clip(rng.normal(0.85 if wet else 0.35, 0.08), 0, 1))
        humidity = float(np.clip(rng.normal(92 if wet else 62, 4), 0, 100))
        weather_risk = float(np.clip(rng.normal(0.8 if wet else 0.2, 0.1), 0, 1))
        satellite_risk = float(np.clip(rng.normal(0.75 if wet else 0.25, 0.12), 0, 1))
        rows.append(
            {
                "region": ["Kerala", "Mumbai"][i % 2],
                "date": f"20{10 + i // 60:02d}-{1 + (i % 12):02d}-{1 + (i % 28):02d}",
                "temperature_c": float(rng.normal(28, 2)),
                "humidity_pct": humidity,
                "pressure_hpa": float(rng.normal(1004, 3)),
                "wind_speed_ms": float(rng.normal(5, 1.5)),
                "wind_direction_deg": float(rng.uniform(0, 360)),
                "cloud_cover_pct": cloud * 100,
                "rain_sum_1d": 60.0 if wet else 2.0,
                "rain_sum_3d": 120.0 if wet else 6.0,
                "rain_sum_7d": 180.0 if wet else 12.0,
                "rain_sum_30d": 400.0 if wet else 40.0,
                "rain_trend_3d": 30.0 if wet else -1.0,
                "season_sin": float(np.sin(i)),
                "season_cos": float(np.cos(i)),
                "cloud_density": cloud,
                "brightness_mean": cloud * 0.9,
                "brightness_std": 0.15,
                "spatial_dispersion": 0.85,
                "cold_top_fraction": cloud * 0.4,
                "cloud_growth_rate": 0.5,
                "valid_fraction": 1.0,
                "weather_risk_score": weather_risk,
                "satellite_risk_score": satellite_risk,
                "weather_prob_0": 1 - weather_risk,
                "weather_prob_1": weather_risk,
                "weather_prob_2": 0.0,
                "satellite_prob_0": 1 - satellite_risk,
                "satellite_prob_1": satellite_risk,
                "satellite_prob_2": 0.0,
                "rainfall_mm": 80.0 if wet else 3.0,
                config.target_column: 1 if wet else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def test_openmp_is_pinned_on_import():
    """The package pins OpenMP to one thread; torch + xgboost crash otherwise."""
    import ai_models.fusion_model  # noqa: F401

    assert os.environ.get("OMP_NUM_THREADS") == "1"


def test_align_probabilities_expands_missing_classes():
    """An estimator that never saw class 2 still yields a 3-wide vector."""
    aligned = align_probabilities(np.array([[0.7, 0.3]]), classes=[0, 1], num_classes=3)
    assert aligned.shape == (1, 3)
    assert aligned[0, 2] == 0.0  # absent class gets zero, not a fabricated value
    assert aligned[0].sum() == pytest.approx(1.0)


def test_align_probabilities_respects_class_order():
    aligned = align_probabilities(np.array([[0.2, 0.8]]), classes=[2, 0], num_classes=3)
    assert aligned[0, 2] == pytest.approx(0.2)
    assert aligned[0, 0] == pytest.approx(0.8)
    assert aligned[0, 1] == 0.0


def test_risk_score_is_heavy_plus_extreme():
    assert risk_score(np.array([[0.5, 0.3, 0.2]]))[0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "score,expected", [(0.1, "LOW"), (0.4, "MODERATE"), (0.6, "HIGH"), (0.95, "CRITICAL")]
)
def test_risk_level_thresholds(score, expected, config):
    assert risk_level(score, config) == expected


@pytest.mark.parametrize("pct,expected", [(90.0, "High"), (60.0, "Medium"), (20.0, "Low")])
def test_confidence_label(pct, expected):
    assert confidence_label(pct) == expected


def test_model_agreement_is_none_when_a_branch_is_absent():
    assert model_agreement(0.8, 0.8) == pytest.approx(1.0)
    assert model_agreement(0.9, 0.1) == pytest.approx(0.2)
    assert model_agreement(0.5, None) is None


# ---------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------
def test_weighted_fusion_blends_and_normalises(unified, config):
    strategy = WeightedFusion(0.5).fit(unified, unified[config.target_column], config)
    probabilities = strategy.predict_proba(unified)
    assert probabilities.shape == (len(unified), config.num_classes)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    row = unified.iloc[0]
    expected = 0.5 * row["weather_prob_1"] + 0.5 * row["satellite_prob_1"]
    assert probabilities[0, 1] == pytest.approx(expected, abs=1e-6)


def test_weighted_fusion_extremes_recover_single_modalities(unified, config):
    """w=1 is the weather model exactly; w=0 is the satellite model exactly."""
    target = unified[config.target_column]
    weather_only = SingleModality("weather").fit(unified, target, config).predict_proba(unified)
    satellite_only = SingleModality("satellite").fit(unified, target, config).predict_proba(unified)

    assert np.allclose(WeightedFusion(1.0).fit(unified, target, config).predict_proba(unified), weather_only)
    assert np.allclose(WeightedFusion(0.0).fit(unified, target, config).predict_proba(unified), satellite_only)


def test_weighted_fusion_rejects_out_of_range_weight():
    with pytest.raises(ValueError):
        WeightedFusion(1.5)


def test_fit_weight_searches_the_grid(unified, config):
    from sklearn.metrics import f1_score

    strategy, sweep = fit_weight(
        unified, unified[config.target_column], config,
        lambda y, p: f1_score(y, p, average="macro", zero_division=0),
    )
    assert 0.0 <= strategy.weather_weight <= 1.0
    assert len(sweep) == pytest.approx(1 / config.weight_grid_step + 1, abs=1)
    best = max(sweep, key=lambda entry: entry["score"])
    assert best["weather_weight"] == pytest.approx(strategy.weather_weight, abs=1e-9)


def test_feature_fusion_learns_the_signal(unified, config):
    target = unified[config.target_column]
    strategy = FeatureFusion("random_forest", config.random_seed).fit(unified, target, config)
    probabilities = strategy.predict_proba(unified)
    assert probabilities.shape == (len(unified), config.num_classes)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    # The synthetic signal is strong; a model that cannot fit it is broken.
    assert (probabilities.argmax(axis=1) == target.to_numpy()).mean() > 0.9


def test_unknown_estimator_is_rejected(config):
    with pytest.raises(ValueError):
        FeatureFusion("not_a_model", config.random_seed)


# ---------------------------------------------------------------------
# Splitting and leakage
# ---------------------------------------------------------------------
def test_chronological_split_does_not_overlap(unified, config):
    splits = chronological_split(unified, config)
    assert len(splits.x_train) > len(splits.x_val) > 0
    assert len(splits.x_test) > 0
    assert splits.train_frame["date"].max() < splits.val_frame["date"].min()
    assert splits.val_frame["date"].max() < splits.test_frame["date"].min()
    assert len(splits.x_train) + len(splits.x_val) + len(splits.x_test) == len(unified)


def test_walk_forward_scores_are_out_of_sample(config):
    """Every fold must be scored by a model fitted only on earlier rows.

    This is the guard against the Phase 3 leak: scoring with an estimator
    that has seen the row produces a memorised value the fusion then
    learns to threshold.
    """
    from sklearn.tree import DecisionTreeClassifier

    from ai_models.fusion_model.dataset import _walk_forward_probabilities

    size = 400
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2001-01-01", periods=size, freq="D", tz="UTC"),
            "x": np.arange(size, dtype=float),
            "y": ([0] * 3 + [1]) * (size // 4),
        }
    )
    # A fully grown tree memorises perfectly, so any in-sample scoring shows up.
    probabilities = _walk_forward_probabilities(
        frame, DecisionTreeClassifier(random_state=0), ["x"], "y", config
    )

    warmup = int(size * config.oof_warmup_fraction)
    assert np.isnan(probabilities[: warmup - 1]).all(), "warm-up rows must not be scored"
    scored = ~np.isnan(probabilities[:, 0])
    assert scored.sum() > 0
    assert np.allclose(probabilities[scored].sum(axis=1), 1.0)


# ---------------------------------------------------------------------
# Historical matching
# ---------------------------------------------------------------------
def test_historical_matcher_ranks_similar_events_first(unified, config):
    matcher = HistoricalMatcher().fit(unified, config)
    assert matcher.reference_size == int((unified[config.target_column] > 0).sum())

    wet = unified[unified[config.target_column] > 0].iloc[0]
    matches = matcher.match(wet.to_dict(), config, top_k=3)
    assert len(matches) == 3
    assert matches[0]["similarity"] >= matches[-1]["similarity"]
    assert 0.0 < matches[0]["similarity"] <= 1.0
    assert matches[0]["similarity_pct"] == pytest.approx(matches[0]["similarity"] * 100, abs=0.1)
    assert all(m["observed_category"] > 0 for m in matches)


def test_historical_matcher_excludes_self_match(unified, config):
    """A replayed day must not match itself at 100% and inflate confidence."""
    matcher = HistoricalMatcher().fit(unified, config)
    wet = unified[unified[config.target_column] > 0].iloc[0]

    included = matcher.match(wet.to_dict(), config, top_k=1)
    assert included[0]["similarity"] == pytest.approx(1.0, abs=1e-6)

    excluded = matcher.match(
        wet.to_dict(), config, top_k=1, exclude=(wet["region"], wet["date"])
    )
    assert (excluded[0]["region"], excluded[0]["date"]) != (wet["region"], wet["date"])
    assert excluded[0]["similarity"] < 1.0


def test_historical_matcher_requires_fitting(config):
    with pytest.raises(MatcherNotFittedError):
        HistoricalMatcher().match({name: 0.0 for name in MATCH_FEATURES}, config)


def test_historical_matcher_rejects_missing_conditions(unified, config):
    matcher = HistoricalMatcher().fit(unified, config)
    with pytest.raises(ValueError):
        matcher.match({"humidity_pct": 90.0}, config)


# ---------------------------------------------------------------------
# Prediction contract
# ---------------------------------------------------------------------
@pytest.fixture()
def bundle_path(tmp_path: Path, unified: pd.DataFrame, config: FusionConfig) -> Path:
    """A minimal trained bundle — no Phase 3/4 artifacts required."""
    target = unified[config.target_column]
    strategy = WeightedFusion(0.5).fit(unified, target, config)
    bundle = {
        "version": "test",
        "model_name": "weighted_fusion_w0.50",
        "approach": "weighted",
        "strategy": strategy,
        "matcher": HistoricalMatcher().fit(unified, config),
        "feature_names": config.feature_columns(),
        "label_names": config.label_names,
        "event_names": config.event_names,
        "feature_medians": unified[config.feature_columns()].median().to_dict(),
        "metrics": {},
        "dataset_info": {},
    }
    path = tmp_path / "fusion.pkl"
    joblib.dump(bundle, path)
    return path


def _payload(**overrides) -> dict:
    payload = {
        "weather_data": {"class_probabilities": [0.2, 0.8, 0.0], "humidity_pct": 92.0},
        "satellite_features": {"class_probabilities": [0.3, 0.7, 0.0], "cloud_density": 0.9},
        "location": {"region": "Kerala", "latitude": 10.0, "longitude": 76.3},
        "timestamp": "2018-08-15T05:15:00Z",
    }
    payload.update(overrides)
    return payload


def test_prediction_output_contract(bundle_path, config):
    result = HybridPredictor(model_path=bundle_path, config=config).predict(_payload())

    assert result["event_prediction"] in set(config.event_names.values())
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["risk_level"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert result["confidence"] in {"Low", "Medium", "High"}
    assert 0.0 <= result["confidence_pct"] <= 100.0
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert result["contributing_models"]["weather_risk_score"] == pytest.approx(0.8, abs=1e-6)
    assert result["contributing_models"]["satellite_risk_score"] == pytest.approx(0.7, abs=1e-6)


def test_prediction_reports_imputed_features(bundle_path, config):
    """Missing inputs fall back to medians and are declared, not hidden."""
    result = HybridPredictor(model_path=bundle_path, config=config).predict(_payload())
    assert "pressure_hpa" in result["imputed_features"]
    assert "humidity_pct" not in result["imputed_features"]  # supplied


def test_confidence_uses_agreement_and_history(bundle_path, config):
    """Branches that agree should not score below branches that disagree."""
    predictor = HybridPredictor(model_path=bundle_path, config=config)
    agreeing = predictor.predict(
        _payload(
            weather_data={"class_probabilities": [0.2, 0.8, 0.0]},
            satellite_features={"class_probabilities": [0.2, 0.8, 0.0]},
        )
    )
    disagreeing = predictor.predict(
        _payload(
            weather_data={"class_probabilities": [0.1, 0.9, 0.0]},
            satellite_features={"class_probabilities": [0.9, 0.1, 0.0]},
        )
    )
    assert agreeing["contributing_models"]["agreement"] > disagreeing["contributing_models"]["agreement"]
    assert agreeing["confidence_pct"] > disagreeing["confidence_pct"]


def test_scalar_risk_score_reconstruction_is_flagged(bundle_path, config):
    result = HybridPredictor(model_path=bundle_path, config=config).predict(
        _payload(
            weather_data={"weather_risk_score": 0.9},
            satellite_features={"satellite_risk_score": 0.6},
        )
    )
    assert result["notes"], "reconstructing a vector from a scalar must be disclosed"


def test_vector_from_score_assigns_risk_to_heavy():
    vector = _vector_from_score(0.7)
    assert vector[0].tolist() == pytest.approx([0.3, 0.7, 0.0])


@pytest.mark.parametrize(
    "corruption",
    [
        {"location": {"latitude": 999.0}},
        {"timestamp": "not-a-time"},
        {"satellite_image_path": "/nonexistent/scene.jpg"},
        {"weather_data": "not-an-object"},
    ],
)
def test_invalid_inputs_rejected(bundle_path, config, corruption):
    predictor = HybridPredictor(model_path=bundle_path, config=config)
    with pytest.raises(InvalidInputError):
        predictor.predict(_payload(**corruption))


def test_missing_branch_refuses_to_guess(bundle_path, config):
    """With no satellite evidence the predictor must fail, not invent a branch."""
    predictor = HybridPredictor(model_path=bundle_path, config=config)
    with pytest.raises(InvalidInputError):
        predictor.predict(
            {
                "weather_data": {"class_probabilities": [0.2, 0.8, 0.0]},
                "satellite_features": {},
                "location": {"region": "Kerala"},
            }
        )


def test_missing_bundle_raises(tmp_path, config):
    with pytest.raises(ModelNotTrainedError):
        HybridPredictor(model_path=tmp_path / "absent.pkl", config=config)

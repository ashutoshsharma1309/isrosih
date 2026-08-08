import pytest

from ai_models.baseline.model import BaselineRainfallModel, load_bundle
from ai_models.baseline.predict import (
    InvalidInputError,
    ModelNotTrainedError,
    RainfallPredictor,
)

VALID_INPUT = {
    "latitude": 10.0,
    "longitude": 76.3,
    "temperature_c": 26.0,
    "humidity_pct": 92.0,
    "pressure_hpa": 1001.0,
    "wind_speed_ms": 8.0,
    "cloud_cover_pct": 95.0,
}


def test_bundle_loads_and_contains_contract_keys(trained_bundle_path):
    bundle = load_bundle(trained_bundle_path)
    assert bundle["model_name"] == "random_forest"
    assert len(bundle["feature_names"]) > 0
    assert set(bundle["label_names"]) == {0, 1, 2}


def test_prediction_output_format(baseline_config, trained_bundle_path):
    predictor = RainfallPredictor(model_path=trained_bundle_path, config=baseline_config)
    result = predictor.predict(VALID_INPUT)

    assert result["prediction"] in {"Normal Rainfall", "Heavy Rainfall", "Extreme Rainfall"}
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["confidence"] in {"Low", "Medium", "High"}
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-3
    assert result["category"] in {0, 1, 2}
    # History wasn't provided → the response must say what was assumed.
    assert any(f.startswith("rain_") for f in result["assumed_features"])


def test_full_history_input_assumes_nothing(baseline_config, trained_bundle_path):
    predictor = RainfallPredictor(model_path=trained_bundle_path, config=baseline_config)
    result = predictor.predict(
        {
            **VALID_INPUT,
            "wind_direction_deg": 240.0,
            "recent_rainfall_mm_1d": 30.0,
            "recent_rainfall_mm_3d": 80.0,
            "recent_rainfall_mm_7d": 150.0,
            "recent_rainfall_mm_30d": 400.0,
        }
    )
    assert result["assumed_features"] == []


@pytest.mark.parametrize(
    "corruption",
    [
        {"latitude": 999.0},                 # out of range
        {"humidity_pct": 150.0},             # out of range
        {"temperature_c": "warm"},           # non-numeric
        {"pressure_hpa": None},              # required but null
        {"latitude": float("nan")},          # NaN
    ],
)
def test_invalid_inputs_rejected(baseline_config, trained_bundle_path, corruption):
    predictor = RainfallPredictor(model_path=trained_bundle_path, config=baseline_config)
    with pytest.raises(InvalidInputError):
        predictor.predict({**VALID_INPUT, **corruption})


def test_missing_required_field_rejected(baseline_config, trained_bundle_path):
    payload = dict(VALID_INPUT)
    del payload["humidity_pct"]
    predictor = RainfallPredictor(model_path=trained_bundle_path, config=baseline_config)
    with pytest.raises(InvalidInputError):
        predictor.predict(payload)


def test_missing_model_raises_not_trained(baseline_config):
    with pytest.raises(ModelNotTrainedError):
        RainfallPredictor(
            model_path=baseline_config.saved_models_dir / "does_not_exist.pkl",
            config=baseline_config,
        )


def test_rainfall_model_interface_adapter(baseline_config, trained_bundle_path):
    """BaselineRainfallModel must satisfy the Phase 1 RainfallModel contract."""
    model = BaselineRainfallModel()
    model.load(trained_bundle_path.parent)

    output = model.predict({"humidity_pct": 90.0, "latitude": 10.0, "longitude": 76.3}, None)
    assert 0.0 <= output.probability <= 1.0
    assert output.risk_level in {"low", "heavy", "extreme"}
    assert 0.0 <= output.confidence <= 1.0
    assert "class_probabilities" in output.explanation_context
    assert model.feature_names()

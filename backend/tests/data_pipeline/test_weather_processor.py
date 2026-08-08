import numpy as np
import pandas as pd
import pytest

from data_pipeline.preprocessors.weather_processor import WeatherDataError, WeatherProcessor


def test_process_full_pipeline(config, weather_frame):
    processor = WeatherProcessor(config)
    result, summary = processor.process(weather_frame)

    assert summary.input_records == 24
    assert summary.interpolated_values >= 2  # the humidity gap was filled
    assert result["humidity_pct"].isna().sum() == 0
    assert summary.outliers_flagged >= 1  # the 200 °C spike
    assert bool(result.loc[10, "is_outlier"]) is True
    # Normalized columns exist and stats were persisted for inference reuse.
    assert "temperature_c_norm" in result.columns
    assert (config.paths.metadata / "weather_normalization_stats.json").exists()


def test_normalize_with_saved_stats_is_reproducible(config, weather_frame):
    processor = WeatherProcessor(config)
    fitted, stats = processor.normalize(weather_frame)
    reapplied, _ = processor.normalize(weather_frame, stats=stats)
    pd.testing.assert_series_equal(
        fitted["temperature_c_norm"], reapplied["temperature_c_norm"]
    )


def test_unparseable_timestamps_are_dropped(config, weather_frame):
    weather_frame["timestamp"] = weather_frame["timestamp"].astype(str)
    weather_frame.loc[0, "timestamp"] = "garbage"
    result, _ = WeatherProcessor(config).process(weather_frame)
    assert len(result) == 23
    assert result["timestamp"].dt.tz is not None  # tz-aware UTC


def test_missing_required_columns_raise(config):
    with pytest.raises(WeatherDataError):
        WeatherProcessor(config).process(pd.DataFrame({"temperature_c": [1.0]}))


def test_long_gaps_are_not_invented(config, weather_frame):
    """Gaps longer than max_interpolation_gap must stay missing."""
    gap = config.weather.max_interpolation_gap + 4
    weather_frame.loc[8 : 8 + gap, "pressure_hpa"] = np.nan
    result, summary = WeatherProcessor(config).process(weather_frame)
    assert summary.remaining_missing > 0
    assert result["pressure_hpa"].isna().sum() > 0

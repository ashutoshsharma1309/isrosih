import pandas as pd
import pytest

from data_pipeline.merger import DatasetMerger, MergeError
from data_pipeline.preprocessors.rainfall_processor import RainfallProcessor
from data_pipeline.preprocessors.weather_processor import WeatherProcessor


def _processed_inputs(config, weather_frame, rainfall_frame):
    weather, _ = WeatherProcessor(config).process(weather_frame)
    rainfall, _ = RainfallProcessor(config).process(rainfall_frame)
    return weather, rainfall


def test_merge_joins_only_true_overlaps(config, weather_frame, rainfall_frame):
    weather, rainfall = _processed_inputs(config, weather_frame, rainfall_frame)
    merged, summary = DatasetMerger(config).merge(weather, rainfall)

    assert summary.merged_records == 24  # same hours, same rounded grid point
    assert {"timestamp", "latitude", "longitude", "rainfall_mm",
            "rainfall_category", "satellite_image_path"} <= set(merged.columns)
    assert merged["satellite_image_path"].isna().all()  # none provided
    assert summary.dataset_path.exists()


def test_merge_attaches_nearest_satellite_image_within_window(
    config, weather_frame, rainfall_frame
):
    weather, rainfall = _processed_inputs(config, weather_frame, rainfall_frame)
    index = pd.DataFrame(
        {
            "captured_at": ["2024-07-01T03:10:00Z", "2024-07-01T22:40:00Z"],
            "output_path": ["data/processed/images/T/a.npy", "data/processed/images/T/b.npy"],
        }
    )
    merged, summary = DatasetMerger(config).merge(weather, rainfall, index)

    assert summary.with_satellite_image > 0
    linked = merged[merged["satellite_image_path"].notna()]
    at_3am = merged[merged["timestamp"] == pd.Timestamp("2024-07-01T03:00:00Z")]
    assert at_3am["satellite_image_path"].iloc[0].endswith("a.npy")
    # Records far from both capture times stay unlinked (no fabricated links).
    assert len(linked) < len(merged)


def test_merge_with_no_overlap_fails_loudly(config, weather_frame, rainfall_frame):
    weather, rainfall = _processed_inputs(config, weather_frame, rainfall_frame)
    rainfall["timestamp"] = rainfall["timestamp"] + pd.Timedelta(days=365)
    with pytest.raises(MergeError):
        DatasetMerger(config).merge(weather, rainfall)

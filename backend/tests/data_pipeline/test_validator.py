import pandas as pd

from data_pipeline.merger import DatasetMerger
from data_pipeline.preprocessors.rainfall_processor import RainfallProcessor
from data_pipeline.preprocessors.weather_processor import WeatherProcessor
from data_pipeline.validators.data_validator import DataValidator


def _merged_dataset(config, weather_frame, rainfall_frame) -> pd.DataFrame:
    weather, _ = WeatherProcessor(config).process(weather_frame)
    rainfall, _ = RainfallProcessor(config).process(rainfall_frame)
    merged, _ = DatasetMerger(config).merge(weather, rainfall)
    return merged


def test_validator_runs_all_checks_and_writes_report(config, weather_frame, rainfall_frame):
    merged = _merged_dataset(config, weather_frame, rainfall_frame)
    report, report_path = DataValidator(config).validate_and_save(merged)

    check_names = {c.name for c in report.checks}
    assert {"required_columns", "min_records", "missing_values", "duplicates",
            "coordinates", "timestamps", "label_integrity", "class_presence"} <= check_names
    assert report_path.exists()
    text = report_path.read_text()
    assert "VARUNA AI DATA REPORT" in text
    assert "Dataset Status:" in text


def test_small_clean_dataset_fails_only_min_records(config, weather_frame, rainfall_frame):
    """24 clean records: every quality check passes, min_records (100) fails —
    so the dataset is correctly reported NOT READY for training."""
    merged = _merged_dataset(config, weather_frame, rainfall_frame)
    report = DataValidator(config).validate(merged)

    failed = {c.name for c in report.checks if not c.passed}
    assert failed == {"min_records"}
    assert report.ready is False


def test_validator_catches_injected_defects(config, weather_frame, rainfall_frame):
    merged = _merged_dataset(config, weather_frame, rainfall_frame)
    merged.loc[0, "latitude"] = 123.0  # impossible
    merged.loc[1, "rainfall_category"] = 7  # invalid label
    merged = pd.concat([merged, merged.iloc[[2]]], ignore_index=True)  # duplicate

    report = DataValidator(config).validate(merged)
    failed = {c.name for c in report.checks if not c.passed}
    assert {"coordinates", "label_integrity", "duplicates"} <= failed
    assert report.ready is False


def test_missing_columns_short_circuits(config):
    report = DataValidator(config).validate(pd.DataFrame({"foo": [1]}))
    assert report.ready is False
    assert report.checks[0].name == "required_columns"

import pandas as pd
import pytest

from data_pipeline.preprocessors.rainfall_processor import RainfallDataError, RainfallProcessor


def test_categorize_uses_configured_imd_thresholds(config):
    processor = RainfallProcessor(config)
    heavy, extreme = config.rainfall.heavy_mm, config.rainfall.extreme_mm

    assert processor.categorize(0.0) == 0
    assert processor.categorize(heavy - 0.1) == 0
    assert processor.categorize(heavy) == 1
    assert processor.categorize(extreme - 0.1) == 1
    assert processor.categorize(extreme) == 2
    assert processor.categorize(400.0) == 2


def test_categorize_rejects_invalid_values(config):
    processor = RainfallProcessor(config)
    with pytest.raises(ValueError):
        processor.categorize(-5.0)
    with pytest.raises(ValueError):
        processor.categorize(float("nan"))


def test_process_cleans_labels_and_maps_regions(config, rainfall_frame):
    result, summary = RainfallProcessor(config).process(rainfall_frame)

    assert summary.dropped_invalid == 2  # bad timestamp + impossible latitude
    assert summary.output_records == 24
    assert set(result["rainfall_category"].unique()) == {0, 1, 2}
    assert summary.category_counts["normal"] == 8
    assert summary.category_counts["heavy"] == 8
    assert summary.category_counts["extreme"] == 8
    # 10.02N, 76.31E falls inside the configured Kerala bounding box.
    assert (result["region"] == "Kerala").all()
    assert summary.labels_path is not None and summary.labels_path.exists()


def test_missing_required_columns_raise(config):
    with pytest.raises(RainfallDataError):
        RainfallProcessor(config).process(pd.DataFrame({"rainfall_mm": [1.0]}))

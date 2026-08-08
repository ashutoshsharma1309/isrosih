import pandas as pd
import pytest

from ai_models.baseline import data as data_prep
from ai_models.baseline.data import DatasetNotReadyError


def test_load_refuses_without_ready_report(baseline_config, small_dataset):
    baseline_config.validation_report_path.write_text(
        "VARUNA AI DATA REPORT\nDataset Status:       NOT READY\n", encoding="utf-8"
    )
    with pytest.raises(DatasetNotReadyError):
        data_prep.load_dataset(baseline_config)


def test_load_refuses_without_report(baseline_config, small_dataset):
    baseline_config.validation_report_path.unlink()
    with pytest.raises(DatasetNotReadyError):
        data_prep.load_dataset(baseline_config)


def test_supervised_frame_has_all_features_and_no_nans(baseline_config, small_dataset):
    frame = data_prep.build_supervised_frame(small_dataset, baseline_config)
    features = data_prep.feature_columns(baseline_config)

    assert set(features) <= set(frame.columns)
    assert frame[features + [baseline_config.target_column]].isna().sum().sum() == 0
    # 30-day rolling warm-up + final unlabeled day dropped per location.
    assert len(frame) < len(small_dataset)


def test_target_is_next_day_category(baseline_config, small_dataset):
    """The label at row t must equal the raw category at t+1 (no leakage)."""
    frame = data_prep.build_supervised_frame(small_dataset, baseline_config)
    one_location = frame[frame["latitude"] == 10.0].sort_values("timestamp")
    raw = small_dataset[small_dataset["latitude"] == 10.0].set_index(
        pd.to_datetime(small_dataset[small_dataset["latitude"] == 10.0]["timestamp"], utc=True)
    )
    row = one_location.iloc[50]
    next_day = row["timestamp"] + pd.Timedelta(days=1)
    assert row[baseline_config.target_column] == int(raw.loc[next_day, "rainfall_category"])


def test_time_split_is_chronological(baseline_config, small_dataset):
    frame = data_prep.build_supervised_frame(small_dataset, baseline_config)
    splits = data_prep.time_based_split(frame, baseline_config)

    assert len(splits.x_train) > len(splits.x_val) > 0
    assert len(splits.x_test) > 0
    # No temporal overlap between any splits.
    train_max = frame.loc[splits.x_train.index, "timestamp"].max()
    val_min = frame.loc[splits.x_val.index, "timestamp"].min()
    val_max = frame.loc[splits.x_val.index, "timestamp"].max()
    test_min = frame.loc[splits.x_test.index, "timestamp"].min()
    assert train_max < val_min and val_max < test_min

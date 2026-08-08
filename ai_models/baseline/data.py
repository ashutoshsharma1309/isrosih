"""
Dataset preparation for baseline model training.

Loads the Phase 2 training dataset (refusing datasets that are not
validated READY), engineers historical/seasonal features, and produces
time-ordered train/validation/test splits.

Leakage rules enforced here:
- The target is the NEXT day's rainfall category — features only ever
  describe conditions up to and including the current day.
- Rolling rainfall aggregates are computed per location from past rows.
- Splits are chronological; no shuffling across time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_models.baseline.config import BaselineConfig

logger = logging.getLogger(__name__)


class DatasetNotReadyError(RuntimeError):
    """Raised when the Phase 2 validation gate has not passed."""


@dataclass
class SupervisedSplits:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]
    train_range: tuple[str, str]
    test_range: tuple[str, str]


def load_dataset(config: BaselineConfig) -> pd.DataFrame:
    """Load the merged dataset, enforcing the Phase 2 validation gate."""
    report_path = config.validation_report_path
    if not report_path.exists():
        raise DatasetNotReadyError(
            f"No validation report at {report_path}. Run the Phase 2 pipeline "
            "(data_pipeline) before training — models train only on validated data."
        )
    report_text = report_path.read_text(encoding="utf-8")
    if "Dataset Status:       READY" not in report_text:
        raise DatasetNotReadyError(
            "Latest data validation report is NOT READY — refusing to train. "
            f"See {report_path}."
        )
    if not config.dataset_path.exists():
        raise DatasetNotReadyError(f"Training dataset not found: {config.dataset_path}")

    df = pd.read_parquet(config.dataset_path)
    logger.info("Loaded %d records from %s", len(df), config.dataset_path)
    return df


def build_supervised_frame(df: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
    """Engineer features and the next-day target.

    Output columns: configured weather + location features, historical
    rainfall aggregates, seasonal encodings, and `target_category`.
    """
    required = set(config.weather_features) | set(config.location_features) | {
        "timestamp", "rainfall_mm", "rainfall_category",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    frame = df.sort_values(["latitude", "longitude", "timestamp"]).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    grouped = frame.groupby(["latitude", "longitude"], sort=False)

    # --- Historical rainfall features (past-only by construction) -------
    for window in config.rain_windows:
        if window == 1:
            frame[f"rain_sum_{window}d"] = frame["rainfall_mm"]
        else:
            frame[f"rain_sum_{window}d"] = grouped["rainfall_mm"].transform(
                lambda s, w=window: s.rolling(w, min_periods=w).sum()
            )
    # Trend: this 3-day window vs the preceding 3-day window.
    frame["rain_trend_3d"] = frame["rain_sum_3d"] - grouped["rain_sum_3d"].shift(3)

    # --- Seasonal encoding (cyclical, so Dec/Jan are neighbours) --------
    day_of_year = frame["timestamp"].dt.dayofyear
    frame["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # --- Target: next day's category at the same location ---------------
    frame[config.target_column] = grouped["rainfall_category"].shift(-1)

    feature_names = feature_columns(config)
    before = len(frame)
    frame = frame.dropna(subset=feature_names + [config.target_column])
    frame[config.target_column] = frame[config.target_column].astype(int)
    logger.info(
        "Supervised frame: %d rows (%d dropped for rolling warm-up / final day)",
        len(frame), before - len(frame),
    )
    return frame


def feature_columns(config: BaselineConfig) -> list[str]:
    return (
        list(config.weather_features)
        + list(config.location_features)
        + [f"rain_sum_{w}d" for w in config.rain_windows]
        + ["rain_trend_3d", "season_sin", "season_cos"]
    )


def time_based_split(frame: pd.DataFrame, config: BaselineConfig) -> SupervisedSplits:
    """Chronological train/validation/test split on the timestamp column."""
    frame = frame.sort_values("timestamp")
    timestamps = frame["timestamp"]
    t_train_end = timestamps.quantile(config.train_fraction)
    t_val_end = timestamps.quantile(config.train_fraction + config.validation_fraction)

    features = feature_columns(config)
    train = frame[timestamps <= t_train_end]
    val = frame[(timestamps > t_train_end) & (timestamps <= t_val_end)]
    test = frame[timestamps > t_val_end]

    for name, part in (("train", train), ("validation", val), ("test", test)):
        counts = part[config.target_column].value_counts().to_dict()
        logger.info("%s: %d rows, class counts %s", name, len(part), counts)

    return SupervisedSplits(
        x_train=train[features],
        y_train=train[config.target_column],
        x_val=val[features],
        y_val=val[config.target_column],
        x_test=test[features],
        y_test=test[config.target_column],
        feature_names=features,
        train_range=(str(train["timestamp"].min().date()), str(train["timestamp"].max().date())),
        test_range=(str(test["timestamp"].min().date()), str(test["timestamp"].max().date())),
    )

"""
Fixtures for baseline model tests.

A tiny synthetic-but-structured dataset (two locations, ~400 days,
weather correlated with rainfall) trains a small real RandomForest so
tests exercise the actual train/save/load/predict code paths quickly.
Synthetic fixtures are test scaffolding only — the shipped model is
trained exclusively on real pipeline data.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ai_models.baseline import data as data_prep
from ai_models.baseline.config import BaselineConfig
from ai_models.baseline.model import save_bundle


@pytest.fixture()
def baseline_config(tmp_path: Path) -> BaselineConfig:
    return replace(
        BaselineConfig(),
        dataset_path=tmp_path / "training_dataset.parquet",
        validation_report_path=tmp_path / "validation_report.txt",
        saved_models_dir=tmp_path / "saved_models",
        experiments_dir=tmp_path / "experiments",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture()
def small_dataset(baseline_config: BaselineConfig) -> pd.DataFrame:
    rng = np.random.default_rng(seed=11)
    frames = []
    for lat, lon in ((10.0, 76.3), (19.05, 72.85)):
        n = 400
        timestamps = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        humidity = rng.uniform(50, 100, n)
        # Rain loosely follows humidity so models have a real signal.
        rain = np.where(humidity > 85, rng.gamma(2.0, 40.0, n), rng.gamma(1.2, 4.0, n))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "latitude": lat,
                    "longitude": lon,
                    "temperature_c": rng.uniform(22, 34, n),
                    "humidity_pct": humidity,
                    "pressure_hpa": rng.uniform(995, 1015, n),
                    "wind_speed_ms": rng.uniform(0, 15, n),
                    "wind_direction_deg": rng.uniform(0, 360, n),
                    "cloud_cover_pct": rng.uniform(10, 100, n),
                    "rainfall_mm": rain,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["rainfall_category"] = np.select(
        [df["rainfall_mm"] >= 204.4, df["rainfall_mm"] >= 64.5], [2, 1], default=0
    )
    df.to_parquet(baseline_config.dataset_path, index=False)
    baseline_config.validation_report_path.write_text(
        "VARUNA AI DATA REPORT\nDataset Status:       READY\n", encoding="utf-8"
    )
    return df


@pytest.fixture()
def trained_bundle_path(baseline_config: BaselineConfig, small_dataset: pd.DataFrame) -> Path:
    """A real (small) trained bundle in the temp saved_models dir."""
    frame = data_prep.build_supervised_frame(small_dataset, baseline_config)
    splits = data_prep.time_based_split(frame, baseline_config)
    estimator = RandomForestClassifier(
        n_estimators=30, random_state=0, class_weight="balanced", n_jobs=-1
    ).fit(splits.x_train, splits.y_train)
    medians = {name: float(splits.x_train[name].median()) for name in splits.feature_names}
    return save_bundle(
        config=baseline_config,
        model_name="random_forest",
        estimator=estimator,
        feature_names=splits.feature_names,
        feature_medians=medians,
        metrics={},
        dataset_info={"records": len(frame)},
    )

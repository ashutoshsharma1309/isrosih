"""
Shared fixtures for data pipeline tests.

Fixtures build small synthetic inputs in tmp_path — this is standard test
scaffolding for exercising transforms, entirely separate from the product,
which only ever processes real acquired data.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from data_pipeline.config import DataConfig, load_config


@pytest.fixture()
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataConfig:
    """Real repo config, but with all data paths redirected into tmp_path
    so tests never touch the actual data/ tree."""
    cfg = load_config()
    redirected = {
        name: tmp_path / name for name in vars(cfg.paths)
    }
    patched_paths = type(cfg.paths)(**redirected)
    patched_paths.ensure_exist()
    object.__setattr__(cfg, "paths", patched_paths)
    return cfg


@pytest.fixture()
def weather_frame() -> pd.DataFrame:
    """24 hourly records for one grid point, with a short gap and one spike."""
    timestamps = pd.date_range("2024-07-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "latitude": 10.02,
            "longitude": 76.31,
            "temperature_c": np.linspace(24.0, 30.0, 24),
            "humidity_pct": np.linspace(70.0, 95.0, 24),
            "pressure_hpa": np.linspace(1004.0, 998.0, 24),
            "wind_speed_ms": np.linspace(2.0, 8.0, 24),
            "wind_direction_deg": np.linspace(180.0, 270.0, 24),
            "cloud_cover_pct": np.linspace(40.0, 100.0, 24),
            "precipitation_mm": np.linspace(0.0, 12.0, 24),
        }
    )
    df.loc[5:6, "humidity_pct"] = np.nan  # short gap → interpolated
    df.loc[10, "temperature_c"] = 200.0  # physically absurd → outlier flag
    return df


@pytest.fixture()
def rainfall_frame() -> pd.DataFrame:
    """Records spanning all three categories plus invalid rows to clean."""
    timestamps = pd.date_range("2024-07-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "latitude": 10.02,
            "longitude": 76.31,
            "rainfall_mm": np.concatenate(
                [
                    np.linspace(0, 40, 8),      # normal
                    np.linspace(70, 110, 8),    # heavy
                    np.linspace(210, 300, 8),   # extreme
                ]
            ),
        }
    )
    invalid = pd.DataFrame(
        {
            "timestamp": ["not-a-date", "2024-07-01T05:00:00Z"],
            "latitude": [10.0, 99.0],  # second: impossible latitude
            "longitude": [76.3, 76.3],
            "rainfall_mm": [10.0, 10.0],
        }
    )
    return pd.concat([df, invalid], ignore_index=True)


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """Directory with two valid grayscale images and one corrupted file."""
    src = tmp_path / "raw_images"
    src.mkdir()
    rng = np.random.default_rng(seed=7)
    for name in ("scene_a.png", "scene_b.png"):
        pixels = rng.integers(0, 255, size=(300, 400), dtype=np.uint8)
        assert cv2.imwrite(str(src / name), pixels)
    (src / "broken.png").write_bytes(b"this is not a png")
    return src

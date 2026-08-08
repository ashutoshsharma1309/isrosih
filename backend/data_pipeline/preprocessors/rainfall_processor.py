"""
Rainfall data preprocessing and label generation.

Input: a DataFrame of rainfall records (timestamp, latitude, longitude,
rainfall_mm — a 24 h accumulation per data_config.yaml).
Output: cleaned records with an integer `rainfall_category` label:

    0 = normal   (< rainfall.heavy_mm)
    1 = heavy    (>= rainfall.heavy_mm, < rainfall.extreme_mm)
    2 = extreme  (>= rainfall.extreme_mm)

Thresholds follow IMD 24-hour categories and live in config — never in
code. Labels are also written to data/labels/ for training runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.config import DataConfig

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "latitude", "longitude", "rainfall_mm")

CATEGORY_NAMES = {0: "normal", 1: "heavy", 2: "extreme"}


class RainfallDataError(ValueError):
    """Raised when input rainfall data is structurally unusable."""


@dataclass
class RainfallProcessingSummary:
    input_records: int = 0
    output_records: int = 0
    dropped_invalid: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    labels_path: Path | None = None


class RainfallProcessor:
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def categorize(self, rainfall_mm: float) -> int:
        """Map a 24 h rainfall accumulation (mm) to its category label."""
        if rainfall_mm < 0 or not np.isfinite(rainfall_mm):
            raise ValueError(f"Invalid rainfall value: {rainfall_mm}")
        if rainfall_mm >= self.config.rainfall.extreme_mm:
            return 2
        if rainfall_mm >= self.config.rainfall.heavy_mm:
            return 1
        return 0

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Drop structurally invalid records; return (clean_df, dropped_count).

        Removes: unparseable timestamps, out-of-range coordinates,
        negative/non-numeric rainfall, and exact duplicates.
        """
        out = df.copy()
        before = len(out)

        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        out["rainfall_mm"] = pd.to_numeric(out["rainfall_mm"], errors="coerce")
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

        out = out.dropna(subset=["timestamp", "latitude", "longitude", "rainfall_mm"])
        out = out[
            out["latitude"].between(-90, 90)
            & out["longitude"].between(-180, 180)
            & (out["rainfall_mm"] >= 0)
        ]
        out = out.drop_duplicates(subset=["timestamp", "latitude", "longitude"])
        out = out.sort_values("timestamp").reset_index(drop=True)
        return out, before - len(out)

    def map_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach the configured region name containing each record
        (`region` = None for records outside all monitored regions)."""
        out = df.copy()

        def locate(row: pd.Series) -> str | None:
            for region in self.config.regions:
                if region.contains(row["latitude"], row["longitude"]):
                    return region.name
            return None

        out["region"] = out.apply(locate, axis=1)
        return out

    def process(
        self, df: pd.DataFrame, *, write_labels: bool = True
    ) -> tuple[pd.DataFrame, RainfallProcessingSummary]:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise RainfallDataError(f"Rainfall data missing required columns: {missing}")

        summary = RainfallProcessingSummary(input_records=len(df))
        out, dropped = self.clean(df)
        summary.dropped_invalid = dropped
        out = self.map_regions(out)
        out["rainfall_category"] = out["rainfall_mm"].map(self.categorize)

        counts = out["rainfall_category"].value_counts()
        summary.category_counts = {
            CATEGORY_NAMES[cat]: int(counts.get(cat, 0)) for cat in CATEGORY_NAMES
        }
        summary.output_records = len(out)

        if write_labels and not out.empty:
            labels_path = self.config.paths.labels / "rainfall_labels.parquet"
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            out[["timestamp", "latitude", "longitude", "region",
                 "rainfall_mm", "rainfall_category"]].to_parquet(labels_path, index=False)
            summary.labels_path = labels_path

        logger.info(
            "[rainfall] %d→%d records (%d dropped) — categories: %s",
            summary.input_records, summary.output_records, dropped, summary.category_counts,
        )
        return out, summary

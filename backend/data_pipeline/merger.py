"""
Dataset merger — builds the machine-learning-ready training table.

Combines:  processed weather features  +  rainfall labels
           +  (optionally) nearest-in-time satellite image references

Join strategy:
- Timestamps are floored to the hour and coordinates rounded to the
  configured precision, then weather and rainfall are inner-joined on
  (timestamp, latitude, longitude) — only records where BOTH sides truly
  exist survive. No imputation of labels, ever.
- Satellite images are attached per record via nearest-capture-time
  matching (merge_asof) within merger.max_image_time_gap_hours; records
  with no image in the window get satellite_image_path = None rather
  than a fabricated association.

Output: data/processed/datasets/training_dataset.parquet (+ .csv preview)
with the schema documented in docs/dataset_schema.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_pipeline.config import DataConfig

logger = logging.getLogger(__name__)


class MergeError(ValueError):
    """Raised when inputs cannot be merged into a consistent dataset."""


@dataclass
class MergeSummary:
    weather_records: int
    rainfall_records: int
    merged_records: int
    with_satellite_image: int
    dataset_path: Path


class DatasetMerger:
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def merge(
        self,
        weather_df: pd.DataFrame,
        rainfall_df: pd.DataFrame,
        satellite_index: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, MergeSummary]:
        """Produce the training table. See module docstring for strategy."""
        for name, frame, cols in (
            ("weather", weather_df, {"timestamp", "latitude", "longitude"}),
            ("rainfall", rainfall_df, {"timestamp", "latitude", "longitude",
                                       "rainfall_mm", "rainfall_category"}),
        ):
            missing = cols - set(frame.columns)
            if missing:
                raise MergeError(f"{name} data missing columns: {sorted(missing)}")

        weather = self._align_keys(weather_df)
        rainfall = self._align_keys(rainfall_df)

        merged = weather.merge(
            rainfall,
            on=["timestamp", "latitude", "longitude"],
            how="inner",
            suffixes=("", "_rainfall"),
        )
        if merged.empty:
            raise MergeError(
                "Merge produced 0 records — weather and rainfall data do not "
                "overlap in time/space. Check acquisition coverage before proceeding."
            )

        if satellite_index is not None and not satellite_index.empty:
            merged = self._attach_satellite_images(merged, satellite_index)
        else:
            merged["satellite_image_path"] = None

        merged = merged.sort_values("timestamp").reset_index(drop=True)
        dataset_path = self._write(merged)
        summary = MergeSummary(
            weather_records=len(weather_df),
            rainfall_records=len(rainfall_df),
            merged_records=len(merged),
            with_satellite_image=int(merged["satellite_image_path"].notna().sum()),
            dataset_path=dataset_path,
        )
        logger.info(
            "[merger] %d weather + %d rainfall → %d merged (%d with imagery) → %s",
            summary.weather_records, summary.rainfall_records,
            summary.merged_records, summary.with_satellite_image, dataset_path,
        )
        return merged, summary

    # ------------------------------------------------------------------
    def _align_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Floor timestamps to the hour, round coordinates to the join grid."""
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.floor("h")
        precision = self.config.merger.coordinate_precision
        out["latitude"] = out["latitude"].round(precision)
        out["longitude"] = out["longitude"].round(precision)
        return out

    def _attach_satellite_images(
        self, merged: pd.DataFrame, satellite_index: pd.DataFrame
    ) -> pd.DataFrame:
        """Nearest-capture-time image per record, within the configured window.

        satellite_index needs columns: captured_at, output_path.
        """
        required = {"captured_at", "output_path"}
        missing = required - set(satellite_index.columns)
        if missing:
            raise MergeError(f"satellite index missing columns: {sorted(missing)}")

        images = satellite_index.copy()
        images["captured_at"] = pd.to_datetime(images["captured_at"], utc=True)
        images = images.sort_values("captured_at")

        out = merged.sort_values("timestamp")
        out = pd.merge_asof(
            out,
            images[["captured_at", "output_path"]].rename(
                columns={"output_path": "satellite_image_path"}
            ),
            left_on="timestamp",
            right_on="captured_at",
            direction="nearest",
            tolerance=pd.Timedelta(hours=self.config.merger.max_image_time_gap_hours),
        )
        return out.drop(columns=["captured_at"])

    def _write(self, df: pd.DataFrame) -> Path:
        out_dir = self.config.paths.processed_datasets
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = out_dir / "training_dataset.parquet"
        df.to_parquet(dataset_path, index=False)
        # Small CSV head for quick human inspection; parquet is canonical.
        df.head(500).to_csv(out_dir / "training_dataset_preview.csv", index=False)
        return dataset_path

"""
Weather (atmospheric) feature preprocessing.

Input: a DataFrame of weather records with at least a timestamp column,
latitude/longitude, and the feature columns from data_config.yaml.
Output: a cleaned, gap-filled, outlier-flagged and (optionally)
normalized feature table, plus the normalization statistics — which are
persisted so the SAME transform can be applied at inference time.

Steps:
    timestamps → UTC · sort · missing-value interpolation (bounded)
    → outlier flagging (z-score) → z-score normalization (fit or apply)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.config import DataConfig

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "latitude", "longitude")


class WeatherDataError(ValueError):
    """Raised when input weather data is structurally unusable."""


@dataclass
class WeatherProcessingSummary:
    input_records: int = 0
    output_records: int = 0
    interpolated_values: int = 0
    remaining_missing: int = 0
    outliers_flagged: int = 0
    normalization_stats: dict[str, dict[str, float]] = field(default_factory=dict)


class WeatherProcessor:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.features = list(config.weather.features)

    # ------------------------------------------------------------------
    # Individual transforms (each usable standalone, and unit-tested)
    # ------------------------------------------------------------------
    @staticmethod
    def standardize_timestamps(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
        """Parse the timestamp column to timezone-aware UTC datetimes.

        Unparseable timestamps become NaT and are dropped (counted in logs).
        """
        out = df.copy()
        out[column] = pd.to_datetime(out[column], errors="coerce", utc=True)
        invalid = int(out[column].isna().sum())
        if invalid:
            logger.warning("Dropping %d record(s) with unparseable timestamps", invalid)
            out = out.dropna(subset=[column])
        return out.sort_values(column).reset_index(drop=True)

    def handle_missing(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        """Time-interpolate short gaps in feature columns.

        Gaps longer than weather.max_interpolation_gap records are left as
        NaN — inventing long stretches of weather would corrupt training
        data; the validator surfaces them instead.

        Returns (df, values_filled, values_still_missing).
        """
        out = df.copy()
        features = [f for f in self.features if f in out.columns]
        before = int(out[features].isna().sum().sum())
        out[features] = out[features].interpolate(
            method="linear",
            limit=self.config.weather.max_interpolation_gap,
            limit_direction="both",
        )
        after = int(out[features].isna().sum().sum())
        return out, before - after, after

    def flag_outliers(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Add a boolean `is_outlier` column (any feature |z| > threshold).

        Outliers are flagged, not deleted: an extreme-rainfall system must
        not discard the very records that describe extreme conditions.
        Model training decides how to treat them.
        """
        out = df.copy()
        threshold = self.config.weather.outlier_zscore
        flags = pd.Series(False, index=out.index)
        for feature in self.features:
            if feature not in out.columns:
                continue
            series = out[feature]
            std = series.std()
            if not np.isfinite(std) or std == 0:
                continue
            flags |= ((series - series.mean()).abs() / std) > threshold
        out["is_outlier"] = flags
        return out, int(flags.sum())

    def normalize(
        self, df: pd.DataFrame, stats: dict[str, dict[str, float]] | None = None
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """Z-score features into `<feature>_norm` columns.

        Pass `stats` to APPLY a previously fitted transform (inference /
        validation split); omit it to FIT on this data. Raw columns are
        kept so labels and physical values remain inspectable.
        """
        out = df.copy()
        fitted: dict[str, dict[str, float]] = {}
        for feature in self.features:
            if feature not in out.columns:
                continue
            if stats is not None:
                mean, std = stats[feature]["mean"], stats[feature]["std"]
            else:
                mean, std = float(out[feature].mean()), float(out[feature].std())
            safe_std = std if std and np.isfinite(std) and std > 0 else 1.0
            out[f"{feature}_norm"] = (out[feature] - mean) / safe_std
            fitted[feature] = {"mean": mean, "std": safe_std}
        return out, fitted

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process(
        self, df: pd.DataFrame, *, fit_normalization: bool = True
    ) -> tuple[pd.DataFrame, WeatherProcessingSummary]:
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise WeatherDataError(f"Weather data missing required columns: {missing_cols}")

        summary = WeatherProcessingSummary(input_records=len(df))
        out = self.standardize_timestamps(df)
        out = out.drop_duplicates(subset=["timestamp", "latitude", "longitude"])
        out, filled, remaining = self.handle_missing(out)
        summary.interpolated_values = filled
        summary.remaining_missing = remaining
        out, n_outliers = self.flag_outliers(out)
        summary.outliers_flagged = n_outliers
        if fit_normalization:
            out, stats = self.normalize(out)
            summary.normalization_stats = stats
            self._save_stats(stats)
        summary.output_records = len(out)
        logger.info(
            "[weather] %d→%d records, %d interpolated, %d still missing, %d outliers flagged",
            summary.input_records, summary.output_records, filled, remaining, n_outliers,
        )
        return out, summary

    def _save_stats(self, stats: dict[str, dict[str, float]]) -> Path:
        """Persist normalization stats — required to transform inference
        inputs identically to training inputs."""
        path = self.config.paths.metadata / "weather_normalization_stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return path

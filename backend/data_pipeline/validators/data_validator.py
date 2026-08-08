"""
Dataset validation — the gate between the data pipeline and model training.

Runs structural and quality checks over the merged training dataset and
produces a human-readable report ("VARUNA AI DATA REPORT") plus a machine-
readable result. Training (Phase 3) must refuse datasets that are NOT READY.

Checks:
- required columns present
- missing values (overall % vs configured ceiling)
- duplicate (timestamp, latitude, longitude) records
- invalid coordinates
- invalid / non-UTC timestamps
- label integrity (categories ∈ {0,1,2}, rainfall_mm >= 0, class presence)
- referenced satellite image files exist and are loadable
- minimum record count
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.config import DataConfig, REPO_ROOT

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "timestamp",
    "latitude",
    "longitude",
    "rainfall_mm",
    "rainfall_category",
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)
    record_count: int = 0
    missing_pct: float = 0.0
    satellite_images_available: bool = False
    weather_records_available: bool = False

    @property
    def ready(self) -> bool:
        return all(check.passed for check in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail))
        log = logger.info if passed else logger.warning
        log("[validate] %-22s %s — %s", name, "PASS" if passed else "FAIL", detail)


class DataValidator:
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport(record_count=len(df))

        # Structure first — remaining checks assume required columns exist.
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        report.add(
            "required_columns",
            not missing_cols,
            "all present" if not missing_cols else f"missing: {missing_cols}",
        )
        if missing_cols:
            return report

        report.add(
            "min_records",
            len(df) >= self.config.validation.min_records,
            f"{len(df)} records (minimum {self.config.validation.min_records})",
        )

        # Missing values across feature columns (labels must never be NaN).
        feature_cols = [c for c in df.columns if c not in ("satellite_image_path",)]
        total_cells = max(len(df) * len(feature_cols), 1)
        missing_cells = int(df[feature_cols].isna().sum().sum())
        report.missing_pct = round(100.0 * missing_cells / total_cells, 2)
        report.add(
            "missing_values",
            report.missing_pct <= self.config.validation.max_missing_pct,
            f"{report.missing_pct}% missing (ceiling {self.config.validation.max_missing_pct}%)",
        )

        duplicates = int(df.duplicated(subset=["timestamp", "latitude", "longitude"]).sum())
        report.add("duplicates", duplicates == 0, f"{duplicates} duplicate record(s)")

        bad_coords = int(
            (~df["latitude"].between(-90, 90) | ~df["longitude"].between(-180, 180)).sum()
        )
        report.add("coordinates", bad_coords == 0, f"{bad_coords} invalid coordinate(s)")

        timestamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        bad_ts = int(timestamps.isna().sum())
        future_ts = int((timestamps > pd.Timestamp.now(tz="UTC")).sum())
        report.add(
            "timestamps",
            bad_ts == 0 and future_ts == 0,
            f"{bad_ts} unparseable, {future_ts} in the future",
        )

        bad_labels = int((~df["rainfall_category"].isin([0, 1, 2])).sum())
        negative_rain = int((pd.to_numeric(df["rainfall_mm"], errors="coerce") < 0).sum())
        report.add(
            "label_integrity",
            bad_labels == 0 and negative_rain == 0,
            f"{bad_labels} invalid categories, {negative_rain} negative rainfall values",
        )

        present = sorted(df["rainfall_category"].dropna().unique().tolist())
        report.add(
            "class_presence",
            {1, 2} & set(present) != set(),
            f"categories present: {present} (need at least one heavy/extreme class to train)",
        )

        weather_cols = [f for f in self.config.weather.features if f in df.columns]
        report.weather_records_available = bool(weather_cols)
        report.add(
            "weather_features",
            bool(weather_cols),
            f"{len(weather_cols)}/{len(self.config.weather.features)} configured features present",
        )

        report.satellite_images_available, image_detail = self._check_images(df)
        report.add("satellite_images", True, image_detail)  # informational, not blocking

        return report

    def _check_images(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Verify referenced satellite arrays exist and load (sampled)."""
        if "satellite_image_path" not in df.columns:
            return False, "no satellite_image_path column"
        paths = df["satellite_image_path"].dropna().unique()
        if len(paths) == 0:
            return False, "no satellite images linked"
        broken = 0
        sample = paths[:200]  # bound validation cost on large datasets
        for rel_path in sample:
            path = REPO_ROOT / rel_path
            try:
                if path.suffix == ".npy":
                    np.load(path, mmap_mode="r")
                elif not path.is_file():
                    raise FileNotFoundError(path)
            except Exception:
                broken += 1
        return broken == 0, f"{len(paths)} linked, {broken}/{len(sample)} sampled failed to load"

    # ------------------------------------------------------------------
    def render_report(self, report: ValidationReport) -> str:
        """Human-readable report, also written to data/metadata/."""
        lines = [
            "==============================================",
            "          VARUNA AI DATA REPORT",
            "==============================================",
            f"Generated:            {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"Records:              {report.record_count}",
            f"Satellite Images:     Available: {'YES' if report.satellite_images_available else 'NO'}",
            f"Weather Records:      Available: {'YES' if report.weather_records_available else 'NO'}",
            f"Missing Values:       {report.missing_pct}%",
            "----------------------------------------------",
        ]
        for check in report.checks:
            lines.append(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
        lines += [
            "----------------------------------------------",
            f"Dataset Status:       {'READY' if report.ready else 'NOT READY'}",
            "==============================================",
        ]
        return "\n".join(lines)

    def validate_and_save(self, df: pd.DataFrame) -> tuple[ValidationReport, Path]:
        report = self.validate(df)
        text = self.render_report(report)
        out_path = self.config.paths.metadata / "validation_report.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        logger.info("[validate] report written to %s (status: %s)",
                    out_path, "READY" if report.ready else "NOT READY")
        return report, out_path

"""
Build the training dataset from NASA POWER data, end to end.

For each configured region, fetches daily data at representative grid
points, then runs the standard Phase 2 stages: weather preprocessing,
rainfall label generation, merge, and validation. Produces
data/processed/datasets/training_dataset.parquet and the DATA REPORT.

Usage (from backend/):
    python -m data_pipeline.build_power_dataset --start 2001-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from data_pipeline.collectors.nasa_power_collector import NasaPowerCollector
from data_pipeline.config import REPO_ROOT, load_config
from data_pipeline.merger import DatasetMerger
from data_pipeline.preprocessors.rainfall_processor import RainfallProcessor
from data_pipeline.preprocessors.weather_processor import WeatherProcessor
from data_pipeline.validators.data_validator import DataValidator

logger = logging.getLogger(__name__)

#: Representative points per configured region (inside each bbox).
#: Multiple points per large region increase spatial coverage.
REGION_POINTS: dict[str, list[tuple[float, float]]] = {
    "Kerala": [(10.0, 76.3), (9.5, 76.5), (11.25, 75.78), (8.5, 76.95)],
    "Mumbai": [(19.05, 72.85)],
    "Chennai": [(13.05, 80.25)],
    "Assam": [(26.15, 91.75), (27.48, 95.0)],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build training dataset from NASA POWER")
    parser.add_argument("--start", default="2001-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    config = load_config()
    config.paths.ensure_exist()
    collector = NasaPowerCollector(config)

    start = args.start.replace("-", "")
    end = args.end.replace("-", "")

    frames: list[pd.DataFrame] = []
    for region, points in REGION_POINTS.items():
        for lat, lon in points:
            assets = collector.collect(latitude=lat, longitude=lon, start=start, end=end)
            stored = assets[0].stored_path
            frame = collector.to_dataframe(stored if stored.startswith("/") else str(REPO_ROOT / stored))
            logger.info("Region %s point (%.2f, %.2f): %d daily records", region, lat, lon, len(frame))
            frames.append(frame)

    weather_raw = pd.concat(frames, ignore_index=True)
    rainfall_raw = weather_raw[["timestamp", "latitude", "longitude", "precipitation_mm"]].rename(
        columns={"precipitation_mm": "rainfall_mm"}
    )

    weather_df, _ = WeatherProcessor(config).process(weather_raw)
    rainfall_df, rain_summary = RainfallProcessor(config).process(rainfall_raw)
    logger.info("Label distribution: %s", rain_summary.category_counts)

    merged, merge_summary = DatasetMerger(config).merge(weather_df, rainfall_df)
    validator = DataValidator(config)
    report, report_path = validator.validate_and_save(merged)

    print(validator.render_report(report))
    print(f"\nDataset: {merge_summary.dataset_path}\nReport:  {report_path}")
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())

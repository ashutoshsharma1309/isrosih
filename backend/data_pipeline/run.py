"""
Pipeline orchestrator CLI.

Runs the preprocess → merge → validate stages over data already acquired
into data/raw/ (collection itself needs source-specific inputs — order
directories or URL lists — so it is driven via the collector classes or
their docs, not blindly from here).

Usage (from backend/):
    python -m data_pipeline.run --weather-csv path.csv --rainfall-csv path.csv
    python -m data_pipeline.run --satellite-dir ../data/raw/satellite/INSAT-3D_IMG_TIR1 \
        --weather-csv path.csv --rainfall-csv path.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from data_pipeline.config import load_config
from data_pipeline.merger import DatasetMerger
from data_pipeline.preprocessors.rainfall_processor import RainfallProcessor
from data_pipeline.preprocessors.satellite_processor import SatelliteProcessor
from data_pipeline.preprocessors.weather_processor import WeatherProcessor
from data_pipeline.validators.data_validator import DataValidator

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VARUNA AI data pipeline")
    parser.add_argument("--config", type=Path, default=None, help="Path to data_config.yaml")
    parser.add_argument("--weather-csv", type=Path, required=True,
                        help="Weather records (timestamp, latitude, longitude, features)")
    parser.add_argument("--rainfall-csv", type=Path, required=True,
                        help="Rainfall records (timestamp, latitude, longitude, rainfall_mm)")
    parser.add_argument("--satellite-dir", type=Path, default=None,
                        help="Directory of raw satellite images to preprocess and link")
    parser.add_argument("--product", default="INSAT-3D_IMG_TIR1",
                        help="Satellite product name for the processed image index")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    config = load_config(args.config)
    config.paths.ensure_exist()

    # --- Preprocess ---------------------------------------------------
    weather_df, weather_summary = WeatherProcessor(config).process(pd.read_csv(args.weather_csv))
    rainfall_df, rainfall_summary = RainfallProcessor(config).process(pd.read_csv(args.rainfall_csv))

    satellite_index = None
    if args.satellite_dir is not None:
        image_summary = SatelliteProcessor(config).process_directory(args.satellite_dir, args.product)
        if image_summary.index_path and image_summary.index_path.exists():
            satellite_index = pd.read_json(image_summary.index_path, lines=True)
            # Image capture time must come from product metadata during real
            # acquisition; the index carries what the collector recorded.
            if "captured_at" not in satellite_index.columns:
                logger.warning(
                    "Satellite index has no captured_at timestamps — images "
                    "cannot be linked to records and will be skipped."
                )
                satellite_index = None

    # --- Merge + validate ---------------------------------------------
    merged, merge_summary = DatasetMerger(config).merge(weather_df, rainfall_df, satellite_index)
    report, report_path = DataValidator(config).validate_and_save(merged)

    print(DataValidator(config).render_report(report))
    print(f"\nDataset: {merge_summary.dataset_path}\nReport:  {report_path}")
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())

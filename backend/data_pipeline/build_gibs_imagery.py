"""
Build the labeled satellite imagery dataset from NASA GIBS.

Reads the validated Phase 2/3 training dataset, reduces it to region-days
(label = max rainfall category over the region's points that day), keeps
ALL heavy/extreme days plus a stratified sample of normal days, and
downloads MODIS True Color + thermal IR snapshots for each region-day.

Output:
    data/raw/satellite/NASA_GIBS/<layer>/<region>_<date>.jpg
    data/labels/satellite_labels.parquet   (image paths ↔ labels)

Usage (from backend/):
    python -m data_pipeline.build_gibs_imagery --normal-ratio 1.5
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from data_pipeline.collectors.gibs_collector import (
    LAYER_IR_BT,
    LAYER_TRUE_COLOR,
    GibsCollector,
)
from data_pipeline.config import REPO_ROOT, load_config

logger = logging.getLogger(__name__)

#: MODIS Terra archive starts 2000-02-24; all our data (2001+) qualifies.


def build_region_days(dataset_path, normal_ratio: float, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    region_days = (
        df.dropna(subset=["region"])
        .assign(date=lambda d: pd.to_datetime(d["timestamp"], utc=True).dt.strftime("%Y-%m-%d"))
        .groupby(["region", "date"])["rainfall_category"]
        .max()
        .reset_index()
    )
    positives = region_days[region_days["rainfall_category"] > 0]
    normals = region_days[region_days["rainfall_category"] == 0]
    # Stratify the normal sample by region and month so the model can't
    # cheat by associating a region/season with a class.
    n_normal = int(len(positives) * normal_ratio)
    normals = normals.assign(month=normals["date"].str[5:7])
    parts = []
    for _, group in normals.groupby(["region", "month"]):
        share = max(1, round(len(group) / len(normals) * n_normal))
        parts.append(group.sample(n=min(share, len(group)), random_state=seed))
    sample = pd.concat(parts, ignore_index=True).drop(columns=["month"])
    selected = pd.concat([positives, sample], ignore_index=True)
    logger.info(
        "Selected %d region-days (%d positive, %d normal sample)",
        len(selected), len(positives), len(selected) - len(positives),
    )
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download labeled GIBS imagery")
    parser.add_argument("--normal-ratio", type=float, default=1.5,
                        help="Normal region-days per positive region-day")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    config = load_config()
    config.paths.ensure_exist()
    collector = GibsCollector(config)

    selected = build_region_days(
        config.paths.processed_datasets / "training_dataset.parquet", args.normal_ratio, args.seed
    )
    regions = {region.name: region for region in config.regions}
    requests = [
        (regions[row.region], row.date, layer)
        for row in selected.itertuples()
        for layer in (LAYER_TRUE_COLOR, LAYER_IR_BT)
    ]
    logger.info("Downloading %d snapshots (%d region-days x 2 layers)...",
                len(requests), len(selected))
    assets, failures = collector.collect_many(requests, max_workers=args.workers)

    # Index: one row per region-day with both layer paths + label.
    by_key: dict[tuple[str, str], dict] = {}
    for asset in assets:
        key = (asset.extra["region"], asset.extra["date"])
        entry = by_key.setdefault(key, {"region": key[0], "date": key[1]})
        column = "true_color_path" if asset.product == LAYER_TRUE_COLOR else "ir_path"
        entry[column] = asset.stored_path

    index = pd.DataFrame(list(by_key.values())).merge(
        selected.rename(columns={"rainfall_category": "label"}), on=["region", "date"]
    )
    complete = index.dropna(subset=["true_color_path", "ir_path"]) if not index.empty else index
    out_path = config.paths.labels / "satellite_labels.parquet"
    complete.to_parquet(out_path, index=False)

    logger.info("Wrote %d labeled region-days (of %d selected; %d snapshot failures) to %s",
                len(complete), len(selected), len(failures), out_path)
    print(complete["label"].value_counts().sort_index().to_string())
    print(f"\nIndex: {out_path}")
    return 0 if len(complete) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Unified (weather + satellite) dataset assembly for the fusion model.

Pipeline:

    Phase 2 weather records (per point-day)
        → region-day aggregation → past-only rainfall history + season
        → Phase 3 model scored per point, worst case kept per region
    Phase 4 scene features (per region-day) + Phase 4 model risk score
        → inner join on (region, date)
        → target = the *next* day's region rainfall category

Alignment rules enforced here (see config.py for the rationale):

- The target is day T+1 and is only assigned when a record for T+1
  actually exists for that region — never carried across a data gap.
- Rolling rainfall windows use past rows only, so no future information
  reaches a feature.
- Splits are chronological; the join is inner, so a region-day survives
  only when both modalities genuinely observed it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_models.fusion_model.config import FusionConfig
from ai_models.fusion_model.utils import align_probabilities, risk_score

logger = logging.getLogger(__name__)


class FusionDataError(RuntimeError):
    """Raised when an upstream artifact required by the fusion is missing."""


@dataclass
class FusionSplits:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    feature_names: list[str]


# ---------------------------------------------------------------------
# Weather side
# ---------------------------------------------------------------------
def _check_validation_gate(config: FusionConfig) -> None:
    """Refuse to train on data the Phase 2 validator did not pass."""
    report = config.validation_report_path
    if not report.exists():
        raise FusionDataError(
            f"No validation report at {report}. Run the Phase 2 pipeline before fusing."
        )
    if "Dataset Status:       READY" not in report.read_text(encoding="utf-8"):
        raise FusionDataError(f"Data validation report is NOT READY — refusing to train. See {report}.")


def build_region_day_weather(config: FusionConfig) -> pd.DataFrame:
    """Aggregate point-level weather to region-days and engineer history.

    Weather variables are averaged across the region's points; rainfall is
    taken as the regional maximum, matching how Phase 4 labelled a scene
    (worst outcome anywhere in the region that day).
    """
    _check_validation_gate(config)
    if not config.weather_dataset_path.exists():
        raise FusionDataError(f"Weather dataset not found: {config.weather_dataset_path}")

    raw = pd.read_parquet(config.weather_dataset_path)
    missing = ({"region", "timestamp", "rainfall_mm", "rainfall_category"} | set(config.weather_features)) - set(raw.columns)
    if missing:
        raise FusionDataError(f"Weather dataset missing columns: {sorted(missing)}")

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["date"] = raw["timestamp"].dt.strftime("%Y-%m-%d")

    aggregation: dict[str, Any] = {name: "mean" for name in config.weather_features}
    aggregation["rainfall_mm"] = "max"
    aggregation["rainfall_category"] = "max"
    frame = (
        raw.groupby(["region", "date"], as_index=False)
        .agg(aggregation)
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )
    logger.info("Region-day weather: %d rows across %d regions", len(frame), frame["region"].nunique())

    frame["date_dt"] = pd.to_datetime(frame["date"])
    grouped = frame.groupby("region", sort=False)

    # --- Past-only rainfall history -----------------------------------
    for window in config.rain_windows:
        if window == 1:
            frame[f"rain_sum_{window}d"] = frame["rainfall_mm"]
        else:
            frame[f"rain_sum_{window}d"] = grouped["rainfall_mm"].transform(
                lambda s, w=window: s.rolling(w, min_periods=w).sum()
            )
    frame["rain_trend_3d"] = frame["rain_sum_3d"] - grouped["rain_sum_3d"].shift(3)

    # --- Cyclical season ----------------------------------------------
    day_of_year = frame["date_dt"].dt.dayofyear
    frame["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # --- Target: next calendar day, and only if it was actually observed
    frame[config.target_column] = grouped["rainfall_category"].shift(-1)
    next_date = grouped["date_dt"].shift(-1)
    is_consecutive = (next_date - frame["date_dt"]).dt.days == 1
    dropped = int((frame[config.target_column].notna() & ~is_consecutive).sum())
    if dropped:
        logger.info("Dropped %d region-days whose T+1 record was not the next calendar day", dropped)
    frame.loc[~is_consecutive, config.target_column] = np.nan

    return frame


def _walk_forward_probabilities(
    supervised: pd.DataFrame, estimator, features: list[str], target: str, config: FusionConfig
) -> np.ndarray:
    """Out-of-fold Phase 3 probabilities via expanding-window retraining.

    Fold k is scored by a fresh clone of the Phase 3 estimator fitted only
    on days strictly before the fold begins, so no row is ever scored by a
    model that saw it. Rows inside the initial warm-up block are left as
    NaN rather than scored in-sample.
    """
    from sklearn.base import clone

    timestamps = supervised["timestamp"].to_numpy()
    probabilities = np.full((len(supervised), config.num_classes), np.nan, dtype=np.float64)

    ordered = np.sort(np.unique(timestamps))
    warmup_end = ordered[int(len(ordered) * config.oof_warmup_fraction) - 1]
    remaining = ordered[ordered > warmup_end]
    edges = [warmup_end] + [
        remaining[min(int(len(remaining) * (k + 1) / config.oof_folds), len(remaining) - 1)]
        for k in range(config.oof_folds)
    ]

    x_all = supervised[features]
    y_all = supervised[target]
    for fold, (start, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        train_mask = timestamps <= start
        fold_mask = (timestamps > start) & (timestamps <= end)
        if not fold_mask.any() or train_mask.sum() < 100:
            continue
        model = clone(estimator)
        model.fit(x_all[train_mask], y_all[train_mask])
        probabilities[fold_mask] = align_probabilities(
            model.predict_proba(x_all[fold_mask]), model.classes_, config.num_classes
        )
        logger.info(
            "Walk-forward fold %d/%d: trained on %d rows -> scored %d rows (through %s)",
            fold, config.oof_folds, int(train_mask.sum()), int(fold_mask.sum()),
            pd.Timestamp(end).date(),
        )

    scored = int((~np.isnan(probabilities[:, 0])).sum())
    logger.info("Out-of-fold weather scores: %d/%d point-days", scored, len(supervised))
    return probabilities


def add_weather_risk_score(frame: pd.DataFrame, config: FusionConfig) -> pd.DataFrame:
    """Score every point-day with the Phase 3 model; keep the regional worst.

    The Phase 3 bundle was trained on point-level rows, so it is applied at
    that level and aggregated afterwards rather than being fed regional
    averages it never saw. Scores are generated out-of-fold (see
    `_walk_forward_probabilities`) because the shipped Phase 3 artifact has
    memorised most of the fusion period.
    """
    from ai_models.baseline.config import BaselineConfig
    from ai_models.baseline.data import build_supervised_frame, feature_columns
    from ai_models.baseline.model import BUNDLE_FILENAME_TEMPLATE, load_bundle

    baseline_config = BaselineConfig()
    bundle_path = config.saved_models_dir / BUNDLE_FILENAME_TEMPLATE.format(
        version=baseline_config.model_version
    )
    if not bundle_path.exists():
        raise FusionDataError(
            f"Phase 3 model missing: {bundle_path}. Run `python -m ai_models.baseline.train` first."
        )
    bundle = load_bundle(bundle_path)

    points = pd.read_parquet(config.weather_dataset_path)
    points["timestamp"] = pd.to_datetime(points["timestamp"], utc=True)
    supervised = build_supervised_frame(points, baseline_config).sort_values("timestamp")
    feature_names = feature_columns(baseline_config)

    probabilities = _walk_forward_probabilities(
        supervised, bundle["estimator"], feature_names, baseline_config.target_column, config
    )
    probability_columns = [f"weather_prob_{c}" for c in range(config.num_classes)]
    supervised = supervised.assign(
        weather_risk_score=risk_score(probabilities),
        date=supervised["timestamp"].dt.strftime("%Y-%m-%d"),
        **{name: probabilities[:, i] for i, name in enumerate(probability_columns)},
    )
    # Warm-up rows have no out-of-fold score and must not reach the join.
    supervised = supervised.dropna(subset=["weather_risk_score", *probability_columns])
    # Regional worst case: keep the full probability vector of the riskiest
    # point rather than averaging vectors that describe different places.
    worst = supervised.loc[supervised.groupby(["region", "date"])["weather_risk_score"].idxmax()]
    per_region_day = worst[["region", "date", "weather_risk_score", *probability_columns]]
    logger.info(
        "Phase 3 scored %d point-days -> %d region-days (mean risk %.3f)",
        len(supervised), len(per_region_day), float(per_region_day["weather_risk_score"].mean()),
    )
    return frame.merge(per_region_day, on=["region", "date"], how="left")


# ---------------------------------------------------------------------
# Satellite side
# ---------------------------------------------------------------------
def load_satellite_features(config: FusionConfig) -> pd.DataFrame:
    if not config.satellite_features_path.exists():
        raise FusionDataError(
            f"Scene features missing: {config.satellite_features_path}. Run "
            "`python -m ai_models.satellite_model.train` first."
        )
    frame = pd.read_parquet(config.satellite_features_path)
    if frame.empty:
        raise FusionDataError(f"Scene feature table is empty: {config.satellite_features_path}")
    frame = frame.rename(columns={"label": "satellite_same_day_category"})
    frame["date"] = frame["date"].astype(str)
    logger.info("Scene features: %d region-days", len(frame))
    return frame


def add_satellite_risk_score(frame: pd.DataFrame, config: FusionConfig) -> pd.DataFrame:
    """Run the Phase 4 checkpoint over every indexed scene.

    Falls back to leaving the column absent (rather than inventing values)
    if the checkpoint has not been trained yet.
    """
    import torch

    from ai_models.satellite_model.config import REPO_ROOT as SAT_ROOT, SatelliteConfig
    from ai_models.satellite_model.model import CHECKPOINT_TEMPLATE, load_checkpoint
    from ai_models.satellite_model.preprocessing import ImageLoadError, build_transforms, load_rgb

    satellite_config = SatelliteConfig()
    checkpoint_path = config.saved_models_dir / CHECKPOINT_TEMPLATE.format(
        version=satellite_config.model_version
    )
    if not checkpoint_path.exists():
        raise FusionDataError(
            f"Phase 4 checkpoint missing: {checkpoint_path}. Run "
            "`python -m ai_models.satellite_model.train` first."
        )
    if not satellite_config.labels_path.exists():
        raise FusionDataError(f"Satellite label index missing: {satellite_config.labels_path}")

    index = pd.read_parquet(satellite_config.labels_path)
    index["date"] = index["date"].astype(str)
    model, meta = load_checkpoint(checkpoint_path, device="cpu")
    transform = build_transforms(satellite_config, training=False)

    scores: list[dict[str, Any]] = []
    for row in index.itertuples():
        path = Path(row.true_color_path)
        try:
            image = load_rgb(path if path.is_absolute() else SAT_ROOT / path)
        except ImageLoadError as exc:
            logger.warning("Skipping unreadable scene %s %s: %s", row.region, row.date, exc)
            continue
        with torch.no_grad():
            probabilities = torch.softmax(model(transform(image).unsqueeze(0)), dim=1).numpy()
        scores.append(
            {
                "region": row.region,
                "date": row.date,
                "satellite_risk_score": float(risk_score(probabilities)[0]),
                **{
                    f"satellite_prob_{c}": float(probabilities[0, c])
                    for c in range(config.num_classes)
                },
            }
        )

    scored = pd.DataFrame(scores)
    logger.info(
        "Phase 4 (%s) scored %d scenes (mean risk %.3f)",
        meta["model_name"], len(scored), float(scored["satellite_risk_score"].mean()),
    )
    return frame.merge(scored, on=["region", "date"], how="left")


# ---------------------------------------------------------------------
# Assembly + splitting
# ---------------------------------------------------------------------
def build_unified_dataset(config: FusionConfig, *, save: bool = True) -> pd.DataFrame:
    """Join both modalities into the fusion training table."""
    weather = add_weather_risk_score(build_region_day_weather(config), config)
    satellite = add_satellite_risk_score(load_satellite_features(config), config)

    unified = weather.merge(satellite, on=["region", "date"], how="inner")
    logger.info("Joined region-days present in both modalities: %d", len(unified))

    required = (
        config.feature_columns()
        + config.weather_probability_columns()
        + config.satellite_probability_columns()
        + [config.target_column]
    )
    before = len(unified)
    unified = unified.dropna(subset=required)
    logger.info(
        "Fusion dataset: %d rows (%d dropped for missing features/target)", len(unified), before - len(unified)
    )
    if unified.empty:
        raise FusionDataError("No region-days survived the join — cannot train a fusion model.")

    unified[config.target_column] = unified[config.target_column].astype(int)
    unified = unified.sort_values("date").reset_index(drop=True)

    if save:
        config.unified_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        unified.to_parquet(config.unified_dataset_path, index=False)
        logger.info("Wrote unified dataset -> %s", config.unified_dataset_path)
    return unified


def chronological_split(frame: pd.DataFrame, config: FusionConfig) -> FusionSplits:
    """Time-ordered 70/15/15 split — the same discipline as Phases 3 and 4."""
    frame = frame.sort_values("date").reset_index(drop=True)
    dates = sorted(frame["date"].unique())
    train_end = dates[int(len(dates) * config.train_fraction) - 1]
    val_end = dates[int(len(dates) * (config.train_fraction + config.validation_fraction)) - 1]

    train = frame[frame["date"] <= train_end]
    val = frame[(frame["date"] > train_end) & (frame["date"] <= val_end)]
    test = frame[frame["date"] > val_end]

    features = config.feature_columns()
    for name, part in (("train", train), ("val", val), ("test", test)):
        counts = part[config.target_column].value_counts().sort_index().to_dict()
        logger.info("%s: %d region-days, target counts %s", name, len(part), counts)

    return FusionSplits(
        x_train=train[features], y_train=train[config.target_column],
        x_val=val[features], y_val=val[config.target_column],
        x_test=test[features], y_test=test[config.target_column],
        train_frame=train, val_frame=val, test_frame=test,
        feature_names=features,
    )

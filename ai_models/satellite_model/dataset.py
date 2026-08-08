"""
PyTorch dataset over the labeled GIBS imagery index.

The index (data/labels/satellite_labels.parquet, built by
data_pipeline.build_gibs_imagery) maps each region-day to its True Color
scene, IR scene, and real rainfall-category label. Splits are
chronological by date — consistent with Phase 3, so no future scene ever
leaks into training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ai_models.satellite_model.config import REPO_ROOT, SatelliteConfig
from ai_models.satellite_model.preprocessing import build_transforms, load_rgb

logger = logging.getLogger(__name__)


class LabelsNotAvailableError(RuntimeError):
    """Raised when the imagery index is missing or empty."""


def load_labels(config: SatelliteConfig) -> pd.DataFrame:
    if not config.labels_path.exists():
        raise LabelsNotAvailableError(
            f"No satellite label index at {config.labels_path}. Run "
            "`python -m data_pipeline.build_gibs_imagery` first — the model "
            "trains only on real acquired imagery."
        )
    df = pd.read_parquet(config.labels_path)
    if df.empty:
        raise LabelsNotAvailableError(f"Satellite label index is empty: {config.labels_path}")
    # Drop rows whose image files vanished (moved/cleaned data dir).
    resolved = df["true_color_path"].map(lambda p: (REPO_ROOT / p).exists())
    missing = int((~resolved).sum())
    if missing:
        logger.warning("Dropping %d index rows with missing image files", missing)
        df = df[resolved]
    logger.info("Satellite label index: %d scenes, classes %s",
                len(df), df["label"].value_counts().to_dict())
    return df.sort_values("date").reset_index(drop=True)


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(df: pd.DataFrame, config: SatelliteConfig) -> SplitFrames:
    dates = sorted(df["date"].unique())
    train_end = dates[int(len(dates) * config.train_fraction) - 1]
    val_end = dates[int(len(dates) * (config.train_fraction + config.validation_fraction)) - 1]
    splits = SplitFrames(
        train=df[df["date"] <= train_end],
        val=df[(df["date"] > train_end) & (df["date"] <= val_end)],
        test=df[df["date"] > val_end],
    )
    for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        logger.info("%s: %d scenes %s", name, len(part), part["label"].value_counts().to_dict())
    return splits


class SatelliteSceneDataset(Dataset):
    """(normalized image tensor, label) pairs from index rows."""

    def __init__(self, frame: pd.DataFrame, config: SatelliteConfig, *, training: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = build_transforms(config, training=training)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        path = Path(row["true_color_path"])
        image = load_rgb(path if path.is_absolute() else REPO_ROOT / path)
        return self.transform(image), int(row["label"])


def class_weights(frame: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss (rare classes matter)."""
    counts = np.array([max((frame["label"] == c).sum(), 1) for c in range(num_classes)], dtype=np.float64)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)

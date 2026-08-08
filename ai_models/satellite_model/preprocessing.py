"""
Satellite image preprocessing, augmentation, and feature extraction.

Two consumers:
- dataset.py uses the torchvision transform pipelines (train pipeline
  augments: rotation, horizontal flip, brightness/contrast jitter;
  eval pipeline is deterministic).
- Feature extraction computes physically interpretable statistics from
  the True Color + thermal IR scene pair — stored for the Phase 5
  fusion model and surfaced in prediction outputs.

MODIS swath gaps (black wedges where the satellite didn't image) are
excluded from every statistic via a validity mask.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from torchvision import transforms

from ai_models.satellite_model.config import SatelliteConfig

logger = logging.getLogger(__name__)


class ImageLoadError(RuntimeError):
    """Raised when a satellite image cannot be read or decoded."""


def build_transforms(config: SatelliteConfig, *, training: bool) -> transforms.Compose:
    """Torchvision pipeline: PIL/ndarray -> normalized float tensor (C,H,W)."""
    steps: list = [
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
    ]
    if training:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ]
    return transforms.Compose(steps)


def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as RGB uint8 (H, W, 3)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageLoadError(f"Cannot decode satellite image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------
# Interpretable scene features (for fusion + human-readable outputs)
# ---------------------------------------------------------------------
_GAP_THRESHOLD = 0.04   # pixels darker than this in True Color = swath gap
_CLOUD_BRIGHTNESS = 0.55  # True Color brightness above this = cloud
_COLD_TOP_THRESHOLD = 0.35  # IR (warm=bright) below this = cold convective top


def extract_scene_features(
    true_color_path: str | Path,
    ir_path: str | Path | None = None,
    previous_true_color_path: str | Path | None = None,
) -> dict[str, float | None]:
    """Compute interpretable cloud statistics for one scene.

    Returns values in [0, 1] (or None where an input is unavailable —
    never a fabricated number):
      cloud_density        fraction of valid pixels that are cloud
      brightness_mean/std  luminance statistics over valid pixels
      cold_top_fraction    IR: fraction of valid pixels with very cold tops
      spatial_dispersion   evenness of cloud cover across a 4x4 grid
                           (high = organised large system, low = scattered)
      cloud_growth_rate    change in cloud_density vs the previous scene
                           (0.5 = unchanged, >0.5 growing), None w/o history
      valid_fraction       share of the scene actually imaged (swath gaps)
    """
    rgb = load_rgb(true_color_path).astype(np.float32) / 255.0
    gray = rgb.mean(axis=2)
    valid = gray > _GAP_THRESHOLD
    valid_fraction = float(valid.mean())
    if valid_fraction < 0.05:
        raise ImageLoadError(f"Scene is almost entirely swath gap: {true_color_path}")

    cloud_mask = (gray > _CLOUD_BRIGHTNESS) & valid
    cloud_density = float(cloud_mask.sum() / max(valid.sum(), 1))

    # Spatial organisation: per-cell densities over a 4x4 grid.
    h, w = gray.shape
    cells = []
    for i in range(4):
        for j in range(4):
            cell_valid = valid[i * h // 4 : (i + 1) * h // 4, j * w // 4 : (j + 1) * w // 4]
            cell_cloud = cloud_mask[i * h // 4 : (i + 1) * h // 4, j * w // 4 : (j + 1) * w // 4]
            if cell_valid.sum() > 0:
                cells.append(cell_cloud.sum() / cell_valid.sum())
    spatial_dispersion = float(1.0 - np.std(cells)) if cells else None

    features: dict[str, float | None] = {
        "cloud_density": round(cloud_density, 4),
        "brightness_mean": round(float(gray[valid].mean()), 4),
        "brightness_std": round(float(gray[valid].std()), 4),
        "spatial_dispersion": round(spatial_dispersion, 4) if spatial_dispersion is not None else None,
        "valid_fraction": round(valid_fraction, 4),
        "cold_top_fraction": None,
        "cloud_growth_rate": None,
    }

    if ir_path is not None and Path(ir_path).exists():
        ir_gray = load_rgb(ir_path).astype(np.float32).mean(axis=2) / 255.0
        # Reuse the True Color validity mask (same overpass geometry);
        # IR renders warm surfaces bright, cold convective tops dark.
        ir_valid = valid & (ir_gray > 0.0)
        if ir_valid.sum() > 0:
            cold = (ir_gray < _COLD_TOP_THRESHOLD) & ir_valid
            features["cold_top_fraction"] = round(float(cold.sum() / ir_valid.sum()), 4)

    if previous_true_color_path is not None and Path(previous_true_color_path).exists():
        try:
            previous = extract_scene_features(previous_true_color_path)
            delta = cloud_density - (previous["cloud_density"] or 0.0)
            features["cloud_growth_rate"] = round(float(np.clip(0.5 + delta, 0.0, 1.0)), 4)
        except ImageLoadError:
            logger.warning("Previous scene unusable, growth rate unavailable: %s",
                           previous_true_color_path)

    return features

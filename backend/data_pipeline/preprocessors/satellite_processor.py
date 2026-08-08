"""
Satellite image preprocessing.

Turns raw satellite imagery into normalized, fixed-size, tensor-ready
arrays for CNNs / Vision Transformers (Phase 4):

    raw image file → load → corruption check → resize → normalize
    → float32 array saved as .npy + index entry

Output layout:
    data/processed/images/<product>/<name>.npy       (H, W, C) float32
    data/processed/images/<product>_index.jsonl      one entry per image
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from data_pipeline.config import DataConfig, REPO_ROOT

logger = logging.getLogger(__name__)

#: Raster formats OpenCV can decode directly. HDF5/NetCDF scenes (.h5/.nc)
#: need product-specific extraction and are skipped with a warning for now.
_DECODABLE = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class CorruptedImageError(RuntimeError):
    """Raised when an image file cannot be decoded."""


@dataclass
class ImageProcessingSummary:
    processed: int = 0
    corrupted: list[str] = field(default_factory=list)
    skipped_unsupported: list[str] = field(default_factory=list)
    index_path: Path | None = None


class SatelliteProcessor:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.image_size = config.satellite.image_size
        self.normalization = config.satellite.normalization

    # ------------------------------------------------------------------
    # Single-image primitives
    # ------------------------------------------------------------------
    def is_corrupted(self, path: Path) -> bool:
        """True if the file cannot be decoded as an image."""
        if not path.is_file() or path.stat().st_size == 0:
            return True
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return image is None or image.size == 0

    def load_image(self, path: Path) -> np.ndarray:
        """Decode an image file to an (H, W, C) uint8/uint16 array."""
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise CorruptedImageError(f"Cannot decode image: {path}")
        if image.ndim == 2:  # single-channel IR imagery → explicit channel dim
            image = image[:, :, np.newaxis]
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def preprocess_image(self, path: Path) -> np.ndarray:
        """Full single-image pipeline → float32 (size, size, C) in [0, 1]."""
        image = self.load_image(path)
        resized = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        if resized.ndim == 2:  # cv2.resize drops a trailing 1-channel dim
            resized = resized[:, :, np.newaxis]
        return self._normalize(resized.astype(np.float32))

    @staticmethod
    def to_tensor_layout(image: np.ndarray) -> np.ndarray:
        """(H, W, C) → (C, H, W), the layout PyTorch models expect."""
        return np.transpose(image, (2, 0, 1))

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.normalization == "minmax":
            lo, hi = float(image.min()), float(image.max())
            if hi - lo < 1e-8:  # uniform image (e.g. full cloud deck) — avoid /0
                return np.zeros_like(image)
            return (image - lo) / (hi - lo)
        if self.normalization == "zscore":
            std = float(image.std())
            if std < 1e-8:
                return np.zeros_like(image)
            return (image - float(image.mean())) / std
        raise ValueError(f"Unknown normalization mode: {self.normalization}")

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def process_directory(self, source_dir: Path, product: str) -> ImageProcessingSummary:
        """Preprocess every decodable image under source_dir.

        Corrupted files are recorded and skipped — never silently dropped.
        """
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        out_dir = self.config.paths.processed_images / product
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.config.paths.processed_images / f"{product}_index.jsonl"
        summary = ImageProcessingSummary(index_path=index_path)

        with index_path.open("w", encoding="utf-8") as index:
            for path in sorted(source_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in _DECODABLE:
                    if path.suffix.lower() in set(self.config.satellite.allowed_extensions):
                        summary.skipped_unsupported.append(str(path))
                        logger.warning("Unsupported container format (needs extraction): %s", path)
                    continue
                if self.is_corrupted(path):
                    summary.corrupted.append(str(path))
                    logger.warning("Corrupted image skipped: %s", path)
                    continue

                array = self.preprocess_image(path)
                out_path = out_dir / f"{path.stem}.npy"
                np.save(out_path, array)
                index.write(
                    json.dumps(
                        {
                            "source_path": str(path.relative_to(REPO_ROOT))
                            if path.is_relative_to(REPO_ROOT)
                            else str(path),
                            "output_path": str(out_path.relative_to(REPO_ROOT))
                            if out_path.is_relative_to(REPO_ROOT)
                            else str(out_path),
                            "product": product,
                            "shape": list(array.shape),
                            "dtype": str(array.dtype),
                            "normalization": self.normalization,
                        }
                    )
                    + "\n"
                )
                summary.processed += 1

        logger.info(
            "[satellite] processed=%d corrupted=%d unsupported=%d (product=%s)",
            summary.processed, len(summary.corrupted), len(summary.skipped_unsupported), product,
        )
        return summary

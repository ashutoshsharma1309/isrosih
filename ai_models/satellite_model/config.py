"""
Satellite vision model configuration.

All training/evaluation parameters for the Phase 4 satellite image
intelligence module in one reproducible place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ai_models/satellite_model/config.py -> ai_models -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SatelliteConfig:
    # --- Inputs ---
    labels_path: Path = REPO_ROOT / "data/labels/satellite_labels.parquet"
    validation_report_path: Path = REPO_ROOT / "data/metadata/validation_report.txt"

    # --- Outputs ---
    saved_models_dir: Path = REPO_ROOT / "ai_models/saved_models"
    experiments_dir: Path = REPO_ROOT / "ai_models/experiments"
    reports_dir: Path = REPO_ROOT / "reports"
    features_dir: Path = REPO_ROOT / "data/processed/features"

    # --- Image pipeline ---
    image_size: int = 224
    #: ImageNet statistics — required when fine-tuning pretrained backbones.
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # --- Task ---
    num_classes: int = 3
    label_names: dict[int, str] = field(
        default_factory=lambda: {0: "Normal", 1: "Heavy", 2: "Extreme"}
    )

    # --- Split (chronological by date, like Phase 3) ---
    train_fraction: float = 0.70
    validation_fraction: float = 0.15

    # --- Training ---
    batch_size: int = 32
    #: Per-candidate batch size. The pretrained backbones hold far more
    #: activation memory than the small CNN, and the 8 GB development
    #: machine swaps itself to a standstill at batch 32. Missing entries
    #: fall back to `batch_size`.
    batch_size_by_model: dict[str, int] = field(
        default_factory=lambda: {"custom_cnn": 32, "resnet18": 16, "vit_b_16": 8}
    )
    epochs_cnn: int = 12
    epochs_resnet: int = 8
    epochs_vit: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    random_seed: int = 42
    model_version: str = "v1"
    selection_metric: str = "f1_macro"
    good_f1_macro: float = 0.60

    def ensure_output_dirs(self) -> None:
        for path in (self.saved_models_dir, self.experiments_dir, self.reports_dir, self.features_dir):
            path.mkdir(parents=True, exist_ok=True)


def resolve_device() -> str:
    """Best available torch device: Apple MPS > CUDA > CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

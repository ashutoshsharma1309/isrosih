"""
Baseline model configuration.

Everything the training pipeline needs to be reproducible lives here:
feature definitions, split fractions, hyperparameters, and artifact paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ai_models/baseline/config.py -> ai_models -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BaselineConfig:
    # --- Inputs ---
    dataset_path: Path = REPO_ROOT / "data/processed/datasets/training_dataset.parquet"
    validation_report_path: Path = REPO_ROOT / "data/metadata/validation_report.txt"

    # --- Outputs ---
    saved_models_dir: Path = REPO_ROOT / "ai_models/saved_models"
    experiments_dir: Path = REPO_ROOT / "ai_models/experiments"
    reports_dir: Path = REPO_ROOT / "reports"

    # --- Features ---
    # Current-conditions features taken directly from the dataset.
    weather_features: tuple[str, ...] = (
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "wind_speed_ms",
        "wind_direction_deg",
        "cloud_cover_pct",
    )
    location_features: tuple[str, ...] = ("latitude", "longitude")
    # Historical rainfall aggregation windows (days), computed per location
    # from past observations only.
    rain_windows: tuple[int, ...] = (1, 3, 7, 30)

    # --- Task definition ---
    # The model predicts the rainfall category of the NEXT day from
    # conditions known today (see data.build_supervised_frame).
    target_column: str = "target_category"

    # --- Split (time-ordered; no shuffling to avoid leakage) ---
    train_fraction: float = 0.70
    validation_fraction: float = 0.15  # remainder is the test set

    # --- Training ---
    random_seed: int = 42
    model_version: str = "v1"
    #: Model selection metric (computed on the validation split).
    selection_metric: str = "f1_macro"

    # --- Evaluation verdict thresholds (for the model report) ---
    good_f1_macro: float = 0.60

    label_names: dict[int, str] = field(
        default_factory=lambda: {0: "Normal Rainfall", 1: "Heavy Rainfall", 2: "Extreme Rainfall"}
    )

    def ensure_output_dirs(self) -> None:
        for path in (self.saved_models_dir, self.experiments_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)

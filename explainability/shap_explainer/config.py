"""
SHAP explainer configuration.

Impact labels are thresholds applied to *measured* attribution shares, not
a stored ranking: nothing here decides that humidity matters, it only
decides what to call a feature that measurement showed carries 30% of the
attribution mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# explainability/shap_explainer/config.py -> explainability -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ShapConfig:
    # --- Artifacts ---
    saved_models_dir: Path = REPO_ROOT / "ai_models/saved_models"
    fusion_dataset_path: Path = REPO_ROOT / "data/processed/datasets/fusion_dataset.parquet"

    # --- Outputs ---
    reports_dir: Path = REPO_ROOT / "reports/shap"

    # --- Attribution ---
    #: Number of features listed in the ranked explanation.
    top_k: int = 6
    #: Background sample size for the model-agnostic (kernel) path. Kernel
    #: SHAP cost grows with this, so it stays small and is documented.
    background_size: int = 50
    #: Rows sampled for the summary plot describing overall model behaviour.
    summary_sample_size: int = 200

    #: Share of total absolute attribution at which each label starts.
    #: Applied to measured values — see the module docstring.
    impact_thresholds: dict[str, float] = field(
        default_factory=lambda: {"Critical": 0.30, "High": 0.15, "Medium": 0.05}
    )

    #: Human-facing names for the model's feature columns.
    feature_labels: dict[str, str] = field(
        default_factory=lambda: {
            "temperature_c": "Temperature",
            "humidity_pct": "Humidity",
            "pressure_hpa": "Atmospheric pressure",
            "wind_speed_ms": "Wind speed",
            "wind_direction_deg": "Wind direction",
            "cloud_cover_pct": "Cloud cover",
            "rain_sum_1d": "Rainfall, past 24h",
            "rain_sum_3d": "Rainfall, past 3 days",
            "rain_sum_7d": "Rainfall, past 7 days",
            "rain_sum_30d": "Rainfall, past 30 days",
            "rain_trend_3d": "Rainfall trend",
            "season_sin": "Seasonal phase",
            "season_cos": "Seasonal phase",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "cloud_density": "Cloud density",
            "brightness_mean": "Scene brightness",
            "brightness_std": "Brightness variation",
            "spatial_dispersion": "Cloud organisation",
            "cold_top_fraction": "Cold cloud-top fraction",
            "cloud_growth_rate": "Cloud growth rate",
            "valid_fraction": "Usable scene fraction",
            "weather_risk_score": "Weather model risk score",
            "satellite_risk_score": "Satellite model risk score",
        }
    )

    #: Features whose *rise* increases risk, used to phrase direction
    #: ("increase" vs "decrease") in plain language. Pressure is the
    #: meteorologically inverted one: falling pressure precedes storms.
    inverted_features: tuple[str, ...] = ("pressure_hpa",)

    #: Features kept out of the prose narrative. These are geometry and
    #: calendar terms: the model genuinely uses them and their attributions
    #: stay in the table, but "elevated longitude" is not a cause an
    #: operator can act on, and phrasing it as one would be misleading.
    #: The generator discloses how much attribution they absorbed.
    non_narratable_features: tuple[str, ...] = (
        "latitude",
        "longitude",
        "season_sin",
        "season_cos",
        "wind_direction_deg",
    )

    def label_for(self, feature: str) -> str:
        return self.feature_labels.get(feature, feature.replace("_", " "))

    def impact_for(self, share: float) -> str:
        """Map a share of total absolute attribution to an impact label."""
        magnitude = abs(share)
        for label, threshold in sorted(
            self.impact_thresholds.items(), key=lambda kv: -kv[1]
        ):
            if magnitude >= threshold:
                return label
        return "Low"

    def ensure_output_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)

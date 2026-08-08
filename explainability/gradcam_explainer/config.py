"""
Grad-CAM explainer configuration.

Region labelling thresholds operate on the *measured* attribution map:
they decide what counts as a highlighted region, never where the regions
are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# explainability/gradcam_explainer/config.py -> explainability -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GradCamConfig:
    # --- Artifacts ---
    saved_models_dir: Path = REPO_ROOT / "ai_models/saved_models"

    # --- Outputs ---
    reports_dir: Path = REPO_ROOT / "reports/gradcam"

    # --- Overlay rendering ---
    #: Weight of the heatmap against the original scene in the blend.
    overlay_alpha: float = 0.45
    #: OpenCV colormap: red = high influence, blue = low.
    colormap: str = "JET"

    # --- Region extraction ---
    #: Normalised attribution above which a pixel joins a highlighted region.
    region_threshold: float = 0.55
    #: Regions smaller than this share of the scene are dropped as noise.
    min_region_area: float = 0.01
    #: Maximum number of named regions reported.
    max_regions: int = 4

    #: Share of the scene covered by high attribution, mapped to a phrase.
    coverage_labels: dict[str, float] = field(
        default_factory=lambda: {"widespread": 0.30, "substantial": 0.15, "localised": 0.04}
    )

    def ensure_output_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def coverage_label(self, share: float) -> str:
        for label, threshold in sorted(self.coverage_labels.items(), key=lambda kv: -kv[1]):
            if share >= threshold:
                return label
        return "isolated"

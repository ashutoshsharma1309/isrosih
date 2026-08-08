"""
Grad-CAM overlay rendering.

Produces the side-by-side figure the brief asks for: the original scene
next to the same scene with the attribution map blended over it, red where
influence is high and blue where it is low, with the named regions boxed.

Written to `reports/gradcam/`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explainability.gradcam_explainer.config import GradCamConfig
from explainability.gradcam_explainer.gradcam import GradCamExplanation

logger = logging.getLogger(__name__)


def render_heatmap(heatmap: np.ndarray, config: GradCamConfig) -> np.ndarray:
    """Colourise a normalised attribution map as RGB (red high, blue low)."""
    colormap = getattr(cv2, f"COLORMAP_{config.colormap}", cv2.COLORMAP_JET)
    coloured = cv2.applyColorMap((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), colormap)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def blend_overlay(
    image: np.ndarray, heatmap: np.ndarray, config: GradCamConfig
) -> np.ndarray:
    """Blend the colourised attribution over the scene."""
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    coloured = render_heatmap(heatmap, config)
    alpha = float(np.clip(config.overlay_alpha, 0.0, 1.0))
    return cv2.addWeighted(coloured, alpha, image.astype(np.uint8), 1.0 - alpha, 0)


def save_overlay(
    image_path: str | Path,
    explanation: GradCamExplanation,
    config: GradCamConfig,
    *,
    filename: str | None = None,
) -> Path:
    """Render original + overlay side by side with the named regions boxed."""
    from ai_models.satellite_model.preprocessing import load_rgb

    config.ensure_output_dirs()
    image = load_rgb(image_path)
    overlay = blend_overlay(image, explanation.heatmap, config)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6.4))
    left.imshow(image)
    left.set_title(f"Satellite scene — {Path(image_path).stem}", fontsize=10)
    left.axis("off")

    right.imshow(overlay)
    right.set_title(
        f"Grad-CAM ({explanation.model_name}, layer {explanation.target_layer})\n"
        f"satellite risk {explanation.satellite_risk:.2f} — "
        f"{explanation.coverage_label} high-influence coverage "
        f"({explanation.coverage * 100:.1f}% of scene)",
        fontsize=10,
    )
    right.axis("off")

    height, width = image.shape[:2]
    for region in explanation.regions:
        x0, y0, x1, y1 = region.bbox
        rectangle = plt.Rectangle(
            (x0 * width, y0 * height),
            (x1 - x0) * width,
            (y1 - y0) * height,
            fill=False, edgecolor="white", linewidth=1.6,
        )
        right.add_patch(rectangle)
        right.text(
            x0 * width, max(y0 * height - 6, 10), region.name,
            color="white", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.55, pad=1.5, edgecolor="none"),
        )

    fig.suptitle("Red = high influence on the classification · Blue = low", fontsize=9, y=0.03)
    fig.tight_layout()
    path = config.reports_dir / (filename or f"gradcam_{Path(image_path).stem}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grad-CAM overlay -> %s", path)
    return path

"""
Grad-CAM attribution for the Phase 4 satellite model.

Answers "which parts of this scene drove the classification?" by taking
gradients of the predicted class with respect to the final convolutional
feature map, weighting the channels by those gradients, and projecting the
result back onto the image. The target layer is not guessed here — Phase 4
records it in the checkpoint (`gradcam_target_layer`) precisely so this
module does not have to.

Attribution comes from Captum's `LayerGradCam` running on the real
checkpoint. Nothing is synthesised: if the model cannot be loaded or the
scene cannot be read, this raises rather than returning a plausible-looking
map.

One caveat is carried in the output rather than left implicit: when the
Phase 5 fusion assigns the satellite branch weight 0, this heatmap explains
the satellite model's own same-day reading and *not* the fused forecast.
The generator surfaces that wording.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from explainability.gradcam_explainer.config import GradCamConfig

logger = logging.getLogger(__name__)


class GradCamNotAvailableError(RuntimeError):
    """Raised when the satellite checkpoint needed for Grad-CAM is missing."""


class InvalidSceneError(ValueError):
    """Raised when the scene handed to Grad-CAM cannot be used."""


@dataclass
class HighlightedRegion:
    """One contiguous area of the scene the model attended to."""

    name: str
    #: Bounding box in normalised scene coordinates (x0, y0, x1, y1).
    bbox: tuple[float, float, float, float]
    #: Share of the whole scene this region covers.
    area_share: float
    #: Mean normalised attribution inside the region.
    intensity: float
    position: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bbox": [round(float(v), 4) for v in self.bbox],
            "area_share": round(float(self.area_share), 4),
            "intensity": round(float(self.intensity), 4),
            "position": self.position,
        }


@dataclass
class GradCamExplanation:
    """Grad-CAM result for one scene."""

    #: Normalised attribution map in [0, 1], at scene resolution.
    heatmap: np.ndarray
    predicted_category: int
    class_probabilities: dict[str, float]
    satellite_risk: float
    regions: list[HighlightedRegion]
    coverage: float
    coverage_label: str
    target_layer: str
    model_name: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "target_layer": self.target_layer,
            "predicted_category": self.predicted_category,
            "class_probabilities": {k: round(float(v), 4) for k, v in self.class_probabilities.items()},
            "satellite_risk_score": round(float(self.satellite_risk), 4),
            "high_influence_coverage": round(float(self.coverage), 4),
            "coverage_label": self.coverage_label,
            "regions": [r.to_dict() for r in self.regions],
            "notes": self.notes,
        }


class SatelliteGradCam:
    """Grad-CAM over the trained Phase 4 checkpoint."""

    def __init__(self, config: GradCamConfig | None = None) -> None:
        self.config = config or GradCamConfig()
        self._model: Any = None
        self._meta: dict[str, Any] | None = None
        self._transform: Any = None

    # ------------------------------------------------------------------
    def _load(self) -> tuple[Any, dict[str, Any]]:
        if self._model is not None and self._meta is not None:
            return self._model, self._meta

        from ai_models.satellite_model.config import SatelliteConfig
        from ai_models.satellite_model.model import CHECKPOINT_TEMPLATE, load_checkpoint
        from ai_models.satellite_model.preprocessing import build_transforms

        satellite_config = SatelliteConfig()
        path = self.config.saved_models_dir / CHECKPOINT_TEMPLATE.format(
            version=satellite_config.model_version
        )
        if not path.exists():
            raise GradCamNotAvailableError(
                f"No satellite checkpoint at {path}. Run "
                "`python -m ai_models.satellite_model.train` first."
            )
        self._model, self._meta = load_checkpoint(path, device="cpu")
        self._transform = build_transforms(satellite_config, training=False)
        return self._model, self._meta

    @staticmethod
    def _resolve_layer(model: Any, name: str) -> Any:
        """Look up the recorded target layer by dotted name."""
        module = model
        for part in name.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                if not hasattr(module, part):
                    raise GradCamNotAvailableError(
                        f"Checkpoint names target layer '{name}', but '{part}' is not a module "
                        f"of {type(module).__name__}"
                    )
                module = getattr(module, part)
        return module

    # ------------------------------------------------------------------
    def explain(self, image_path: str | Path, target_class: int | None = None) -> GradCamExplanation:
        """Compute the attribution map for one satellite scene.

        Args:
            image_path: True Color scene to explain.
            target_class: class to attribute; defaults to the predicted one.
        """
        import torch
        from captum.attr import LayerGradCam, LayerAttribution

        from ai_models.satellite_model.preprocessing import ImageLoadError, load_rgb

        model, meta = self._load()
        try:
            image = load_rgb(image_path)
        except ImageLoadError as exc:
            raise InvalidSceneError(str(exc)) from exc

        tensor = self._transform(image).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0].numpy()
        predicted = int(np.argmax(probabilities)) if target_class is None else int(target_class)
        if not 0 <= predicted < len(probabilities):
            raise InvalidSceneError(
                f"target_class {predicted} outside the model's {len(probabilities)} classes"
            )

        layer_name = meta["gradcam_target_layer"]
        layer = self._resolve_layer(model, layer_name)

        # Grad-CAM needs gradients, so this runs outside no_grad.
        tensor.requires_grad_(True)
        attributions = LayerGradCam(model, layer).attribute(tensor, target=predicted, relu_attributions=True)
        # The attribution grid is the target layer's spatial resolution (14x14
        # for this CNN). Bilinear upsampling is the standard Grad-CAM
        # presentation — it smooths the same values rather than inventing
        # detail, and avoids a blocky map implying pixel-level precision the
        # attribution does not have.
        upsampled = LayerAttribution.interpolate(
            attributions, (image.shape[0], image.shape[1]), interpolate_mode="bilinear"
        )
        heatmap = upsampled[0].sum(dim=0).detach().numpy()

        span = float(heatmap.max() - heatmap.min())
        heatmap = (heatmap - heatmap.min()) / span if span > 1e-12 else np.zeros_like(heatmap)

        regions, coverage = self._extract_regions(heatmap)
        label_names = {int(k): v for k, v in meta["label_names"].items()}
        notes: list[str] = []
        if span <= 1e-12:
            notes.append(
                "The attribution map is flat: the target layer produced no gradient variation "
                "for this scene, so no region can be singled out."
            )

        return GradCamExplanation(
            heatmap=heatmap,
            predicted_category=predicted,
            class_probabilities={label_names[i]: float(p) for i, p in enumerate(probabilities)},
            satellite_risk=float(probabilities[1:].sum()),
            regions=regions,
            coverage=coverage,
            coverage_label=self.config.coverage_label(coverage),
            target_layer=layer_name,
            model_name=str(meta["model_name"]),
            notes=notes,
        )

    # ------------------------------------------------------------------
    def _extract_regions(self, heatmap: np.ndarray) -> tuple[list[HighlightedRegion], float]:
        """Name the contiguous high-attribution blobs the model attended to."""
        import cv2

        mask = (heatmap >= self.config.region_threshold).astype(np.uint8)
        coverage = float(mask.mean())
        if mask.sum() == 0:
            return [], coverage

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        height, width = heatmap.shape
        scene_area = float(height * width)

        candidates: list[tuple[float, HighlightedRegion]] = []
        for index in range(1, count):  # 0 is the background component
            x, y, w, h, area = stats[index]
            share = area / scene_area
            if share < self.config.min_region_area:
                continue
            blob = heatmap[y : y + h, x : x + w][labels[y : y + h, x : x + w] == index]
            intensity = float(blob.mean()) if blob.size else 0.0
            centre_x, centre_y = centroids[index]
            candidates.append(
                (
                    share * intensity,
                    HighlightedRegion(
                        name="",  # assigned after ranking
                        bbox=(x / width, y / height, (x + w) / width, (y + h) / height),
                        area_share=share,
                        intensity=intensity,
                        position=_describe_position(centre_x / width, centre_y / height),
                    ),
                )
            )

        candidates.sort(key=lambda pair: -pair[0])
        regions = []
        for rank, (_, region) in enumerate(candidates[: self.config.max_regions], start=1):
            region.name = f"cloud_cluster_area_{rank}"
            regions.append(region)
        return regions, coverage


def _describe_position(x: float, y: float) -> str:
    """Plain-language position of a region within the scene."""
    vertical = "northern" if y < 0.36 else "southern" if y > 0.64 else "central"
    horizontal = "western" if x < 0.36 else "eastern" if x > 0.64 else "central"
    if vertical == "central" and horizontal == "central":
        return "centre of the scene"
    if vertical == "central":
        return f"{horizontal} edge of the scene"
    if horizontal == "central":
        return f"{vertical} part of the scene"
    return f"{vertical}-{horizontal} quadrant"

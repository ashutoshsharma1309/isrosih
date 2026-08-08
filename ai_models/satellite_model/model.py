"""
Vision model architectures and the trained-checkpoint format.

Three candidates (compared in train.py):
- "custom_cnn"  — compact 4-block CNN trained from scratch (~1M params)
- "resnet18"    — ImageNet-pretrained ResNet-18, layer4+fc fine-tuned
- "vit_b_16"    — ImageNet-pretrained Vision Transformer, linear probe
                  (frozen backbone; the honest budget-conscious ViT option)

The saved checkpoint bundles the state dict with the architecture name
and metadata so predict.py is fully self-contained.
`SatelliteRainfallModel` adapts a checkpoint to the project-wide
RainfallModel interface for the Phase 5 fusion and Phase 7 backend.
The final convolutional block is exposed by name (`gradcam_target_layer`)
for the Phase 6 Grad-CAM explainer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision import models as tv_models

from ai_models.base import ModelOutput, RainfallModel
from ai_models.satellite_model.config import SatelliteConfig, resolve_device
from ai_models.satellite_model.preprocessing import build_transforms

logger = logging.getLogger(__name__)

CHECKPOINT_TEMPLATE = "satellite_model_{version}.pt"


class CustomCNN(nn.Module):
    """Compact from-scratch CNN baseline for 224x224 RGB scenes."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(name: str, num_classes: int) -> tuple[nn.Module, str]:
    """Instantiate a candidate. Returns (model, gradcam_target_layer)."""
    if name == "custom_cnn":
        return CustomCNN(num_classes), "features.3"
    if name == "resnet18":
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, "layer4"
    if name == "vit_b_16":
        model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.heads = nn.Linear(model.hidden_dim, num_classes)
        # ViT has no conv layer; Phase 6 uses attention rollout instead.
        return model, "encoder.ln"
    raise ValueError(f"Unknown model architecture: {name}")


# ---------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------
def save_checkpoint(
    *,
    config: SatelliteConfig,
    model_name: str,
    model: nn.Module,
    metrics: dict[str, Any],
    dataset_info: dict[str, Any],
    gradcam_target_layer: str,
) -> Path:
    payload = {
        "version": config.model_version,
        "model_name": model_name,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "num_classes": config.num_classes,
        "image_size": config.image_size,
        "label_names": config.label_names,
        "gradcam_target_layer": gradcam_target_layer,
        "metrics": metrics,
        "dataset_info": dataset_info,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    config.saved_models_dir.mkdir(parents=True, exist_ok=True)
    path = config.saved_models_dir / CHECKPOINT_TEMPLATE.format(version=config.model_version)
    torch.save(payload, path)
    logger.info("Saved satellite checkpoint: %s (%s)", path, model_name)
    return path


def load_checkpoint(path: Path, device: str | None = None) -> tuple[nn.Module, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"No trained satellite model at {path}. Run "
            "`python -m ai_models.satellite_model.train` first."
        )
    device = device or resolve_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    model, _ = build_model(payload["model_name"], payload["num_classes"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


# ---------------------------------------------------------------------
# Project-wide interface adapter
# ---------------------------------------------------------------------
class SatelliteRainfallModel(RainfallModel):
    """RainfallModel adapter over a trained satellite checkpoint."""

    _RISK_BY_CATEGORY = {0: "low", 1: "heavy", 2: "extreme"}

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._meta: dict[str, Any] | None = None
        self._device = resolve_device()
        self._transform = None

    def load(self, artifact_dir: Path) -> None:
        version = SatelliteConfig().model_version
        self._model, self._meta = load_checkpoint(
            Path(artifact_dir) / CHECKPOINT_TEMPLATE.format(version=version), self._device
        )
        self._transform = build_transforms(SatelliteConfig(), training=False)
        self.version = self._meta["version"]

    def predict(self, features: dict[str, float] | None, image: np.ndarray | None) -> ModelOutput:
        if self._model is None or self._meta is None:
            raise RuntimeError("Model not loaded — call load() first")
        if image is None:
            raise ValueError("The satellite model requires an image input")

        tensor = self._transform(image.astype(np.uint8)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0].cpu().numpy()
        category = int(np.argmax(probabilities))
        return ModelOutput(
            probability=float(probabilities[1:].sum()),
            risk_level=self._RISK_BY_CATEGORY[category],
            confidence=float(probabilities.max()),
            model_version=f"satellite-{self._meta['model_name']}-{self._meta['version']}",
            explanation_context={
                "class_probabilities": {
                    self._meta["label_names"][i]: float(p) for i, p in enumerate(probabilities)
                },
                "gradcam_target_layer": self._meta["gradcam_target_layer"],
            },
        )

    def feature_names(self) -> list[str]:
        return []  # image model — no tabular features

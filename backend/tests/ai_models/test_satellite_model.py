"""
Satellite model tests.

Synthetic scene images (bright cloud vs dark ground) are generated in
tmp_path as test scaffolding to exercise the real preprocessing, dataset,
training, and prediction code quickly — the shipped checkpoint is trained
exclusively on real NASA GIBS scenes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from ai_models.satellite_model.config import SatelliteConfig
from ai_models.satellite_model.dataset import (
    LabelsNotAvailableError,
    SatelliteSceneDataset,
    chronological_split,
    class_weights,
    load_labels,
)
from ai_models.satellite_model.model import CustomCNN, load_checkpoint, save_checkpoint
from ai_models.satellite_model.predict import (
    InvalidInputError,
    ModelNotTrainedError,
    SatellitePredictor,
)
from ai_models.satellite_model.preprocessing import (
    ImageLoadError,
    build_transforms,
    extract_scene_features,
    load_rgb,
)


@pytest.fixture()
def sat_config(tmp_path: Path) -> SatelliteConfig:
    return replace(
        SatelliteConfig(),
        labels_path=tmp_path / "satellite_labels.parquet",
        saved_models_dir=tmp_path / "saved_models",
        experiments_dir=tmp_path / "experiments",
        reports_dir=tmp_path / "reports",
        features_dir=tmp_path / "features",
    )


def _write_scene(path: Path, cloudy: float, seed: int) -> None:
    """Synthetic 256x256 scene: `cloudy` fraction bright cloud, rest dark."""
    rng = np.random.default_rng(seed)
    image = rng.integers(20, 70, size=(256, 256, 3), dtype=np.uint8)  # dark ground
    n_cloud = int(256 * cloudy)
    image[:, :n_cloud] = rng.integers(190, 255, size=(256, n_cloud, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


@pytest.fixture()
def scene_index(sat_config: SatelliteConfig, tmp_path: Path) -> pd.DataFrame:
    """60 synthetic scenes over consecutive dates; label follows cloudiness."""
    rows = []
    for i in range(60):
        label = i % 3 if i % 5 == 0 else (1 if i % 2 else 0)
        cloudy = {0: 0.25, 1: 0.6, 2: 0.9}[label]
        path = tmp_path / f"scene_{i:03d}.jpg"
        _write_scene(path, cloudy, seed=i)
        rows.append(
            {"region": "Kerala", "date": f"2023-{1 + i // 28:02d}-{1 + i % 28:02d}",
             "true_color_path": str(path), "ir_path": str(path), "label": label}
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(sat_config.labels_path, index=False)
    return frame


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
def test_transforms_produce_normalized_224_tensor(sat_config, tmp_path):
    path = tmp_path / "scene.jpg"
    _write_scene(path, cloudy=0.5, seed=1)
    tensor = build_transforms(sat_config, training=False)(load_rgb(path))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.min() < 0 < tensor.max()  # ImageNet-normalized, not raw [0,1]


def test_augmentation_only_in_training_mode(sat_config, tmp_path):
    path = tmp_path / "scene.jpg"
    _write_scene(path, cloudy=0.5, seed=2)
    image = load_rgb(path)
    eval_transform = build_transforms(sat_config, training=False)
    assert torch.equal(eval_transform(image), eval_transform(image))  # deterministic
    train_transform = build_transforms(sat_config, training=True)
    outputs = [train_transform(image) for _ in range(4)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:])  # stochastic


def test_scene_features_reflect_cloudiness(tmp_path):
    cloudy_path, clear_path = tmp_path / "cloudy.jpg", tmp_path / "clear.jpg"
    _write_scene(cloudy_path, cloudy=0.9, seed=3)
    _write_scene(clear_path, cloudy=0.1, seed=4)

    cloudy = extract_scene_features(cloudy_path, ir_path=None)
    clear = extract_scene_features(clear_path, ir_path=None)
    assert cloudy["cloud_density"] > clear["cloud_density"]
    assert 0.0 <= cloudy["cloud_density"] <= 1.0
    assert cloudy["cold_top_fraction"] is None  # no IR provided → no invention
    growth = extract_scene_features(cloudy_path, previous_true_color_path=clear_path)
    assert growth["cloud_growth_rate"] > 0.5  # clear → cloudy = growing


def test_undecodable_image_raises(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        load_rgb(bad)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
def test_dataset_loading_and_split(sat_config, scene_index):
    index = load_labels(sat_config)
    assert len(index) == 60
    splits = chronological_split(index, sat_config)
    assert len(splits.train) > len(splits.val) > 0 and len(splits.test) > 0
    assert splits.train["date"].max() <= splits.val["date"].min()

    dataset = SatelliteSceneDataset(splits.train, sat_config, training=True)
    tensor, label = dataset[0]
    assert tensor.shape == (3, 224, 224) and label in (0, 1, 2)

    weights = class_weights(splits.train, 3)
    assert weights.shape == (3,) and (weights > 0).all()


def test_missing_labels_index_raises(sat_config):
    with pytest.raises(LabelsNotAvailableError):
        load_labels(sat_config)


# ---------------------------------------------------------------------
# Checkpoint + prediction
# ---------------------------------------------------------------------
@pytest.fixture()
def tiny_checkpoint(sat_config, scene_index) -> Path:
    model = CustomCNN(num_classes=3)
    return save_checkpoint(
        config=sat_config, model_name="custom_cnn", model=model,
        metrics={}, dataset_info={"scenes": len(scene_index)},
        gradcam_target_layer="features.3",
    )


def test_checkpoint_roundtrip(tiny_checkpoint):
    model, meta = load_checkpoint(tiny_checkpoint, device="cpu")
    assert meta["model_name"] == "custom_cnn"
    assert meta["gradcam_target_layer"] == "features.3"
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 3)


def test_prediction_output_format(sat_config, tiny_checkpoint, scene_index):
    predictor = SatellitePredictor(model_path=tiny_checkpoint, config=sat_config)
    result = predictor.predict(
        {
            "satellite_image_path": scene_index.iloc[0]["true_color_path"],
            "ir_image_path": scene_index.iloc[0]["ir_path"],
            "timestamp": "2023-01-01T05:00:00Z",
            "latitude": 10.0,
            "longitude": 76.3,
        }
    )
    assert 0.0 <= result["satellite_risk_score"] <= 1.0
    assert result["cloud_pattern"] in {"Low Risk", "High Risk", "Extreme Risk"}
    assert result["cloud_condition"] in {"Normal", "Heavy", "Extreme"}
    assert result["confidence"] in {"Low", "Medium", "High"}
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-3
    assert result["scene_features"]["cloud_density"] is not None


@pytest.mark.parametrize(
    "corruption",
    [
        {"satellite_image_path": "/nonexistent/scene.jpg"},
        {"satellite_image_path": None},
        {"latitude": 999.0},
        {"timestamp": "not-a-time"},
    ],
)
def test_invalid_inputs_rejected(sat_config, tiny_checkpoint, scene_index, corruption):
    predictor = SatellitePredictor(model_path=tiny_checkpoint, config=sat_config)
    payload = {
        "satellite_image_path": scene_index.iloc[0]["true_color_path"],
        "latitude": 10.0, "longitude": 76.3, **corruption,
    }
    with pytest.raises(InvalidInputError):
        predictor.predict(payload)


def test_missing_checkpoint_raises(sat_config):
    with pytest.raises(ModelNotTrainedError):
        SatellitePredictor(model_path=sat_config.saved_models_dir / "nope.pt", config=sat_config)

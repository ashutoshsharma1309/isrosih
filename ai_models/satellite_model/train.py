"""
Satellite vision model training pipeline.

    load labeled imagery index (real GIBS scenes only)
    → chronological split → train 3 candidates (weighted CE, augmentation)
    → select on validation macro-F1 → single held-out test evaluation
    → checkpoint + SATELLITE AI MODEL REPORT + curves/confusion/samples
    → batch scene-feature extraction for the Phase 5 fusion model

Run (from the repository root):
    backend/.venv/bin/python -m ai_models.satellite_model.train
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ai_models.satellite_model import dataset as data_mod
from ai_models.satellite_model.config import REPO_ROOT, SatelliteConfig, resolve_device
from ai_models.satellite_model.evaluate import (
    collect_predictions,
    compute_metrics,
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_training_curves,
    render_report,
)
from ai_models.satellite_model.model import build_model, save_checkpoint
from ai_models.satellite_model.preprocessing import extract_scene_features

logger = logging.getLogger(__name__)

CANDIDATES = ("custom_cnn", "resnet18", "vit_b_16")


def build_loaders(
    splits: "data_mod.SplitFrames", config: SatelliteConfig, batch_size: int
) -> dict[str, DataLoader]:
    """Loaders at a given batch size; augmentation on the train split only."""
    return {
        "train": DataLoader(
            data_mod.SatelliteSceneDataset(splits.train, config, training=True),
            batch_size=batch_size, shuffle=True, num_workers=0,
        ),
        "val": DataLoader(
            data_mod.SatelliteSceneDataset(splits.val, config, training=False),
            batch_size=batch_size, shuffle=False, num_workers=0,
        ),
        "test": DataLoader(
            data_mod.SatelliteSceneDataset(splits.test, config, training=False),
            batch_size=batch_size, shuffle=False, num_workers=0,
        ),
    }


def train_one(
    name: str,
    config: SatelliteConfig,
    loaders: dict[str, DataLoader],
    weights: torch.Tensor,
    device: str,
) -> tuple[nn.Module, dict[str, list[float]], str]:
    """Train one candidate; returns (model, history, gradcam layer)."""
    model, gradcam_layer = build_model(name, config.num_classes)
    model.to(device)
    epochs = {"custom_cnn": config.epochs_cnn, "resnet18": config.epochs_resnet,
              "vit_b_16": config.epochs_vit}[name]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        losses = []
        for images, targets in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for images, targets in loaders["val"]:
                logits = model(images.to(device))
                val_losses.append(float(criterion(logits, targets.to(device)).item()))
                correct += int((logits.argmax(dim=1).cpu() == targets).sum())
                total += len(targets)

        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(float(np.mean(val_losses)))
        history["val_accuracy"].append(correct / max(total, 1))
        logger.info(
            "[%s] epoch %d/%d  train_loss %.3f  val_loss %.3f  val_acc %.3f  (%.0fs)",
            name, epoch, epochs, history["train_loss"][-1], history["val_loss"][-1],
            history["val_accuracy"][-1], time.time() - started,
        )
    return model, history, gradcam_layer


def extract_features_for_fusion(index: pd.DataFrame, config: SatelliteConfig) -> str:
    """Interpretable scene features for every indexed image → parquet.

    Growth rate uses the previous available scene for the same region
    when one exists; otherwise it stays None (never invented).
    """
    rows: list[dict[str, Any]] = []
    previous_by_region: dict[str, str] = {}
    for row in index.sort_values("date").itertuples():
        try:
            features = extract_scene_features(
                REPO_ROOT / row.true_color_path,
                REPO_ROOT / row.ir_path if isinstance(row.ir_path, str) else None,
                previous_by_region.get(row.region),
            )
        except Exception as exc:
            logger.warning("Feature extraction failed for %s %s: %s", row.region, row.date, exc)
            continue
        previous_by_region[row.region] = REPO_ROOT / row.true_color_path
        rows.append({"region": row.region, "date": row.date, "label": int(row.label), **features})

    frame = pd.DataFrame(rows)
    path = config.features_dir / "satellite_features.parquet"
    frame.to_parquet(path, index=False)
    logger.info("Scene features for fusion: %d rows -> %s", len(frame), path)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    config = SatelliteConfig()
    config.ensure_output_dirs()
    device = resolve_device()
    torch.manual_seed(config.random_seed)
    logger.info("Training on device: %s", device)

    index = data_mod.load_labels(config)
    splits = data_mod.chronological_split(index, config)
    weights = data_mod.class_weights(splits.train, config.num_classes)

    histories: dict[str, dict[str, list[float]]] = {}
    validation_metrics: dict[str, dict[str, Any]] = {}
    fitted: dict[str, tuple[nn.Module, str]] = {}
    for name in CANDIDATES:
        batch_size = config.batch_size_by_model.get(name, config.batch_size)
        candidate_loaders = build_loaders(splits, config, batch_size)
        logger.info("Training %s at batch size %d", name, batch_size)
        try:
            model, history, gradcam_layer = train_one(
                name, config, candidate_loaders, weights, device
            )
        except Exception as exc:
            logger.warning("Candidate %s failed to train (%s) — skipping", name, exc)
            continue
        histories[name] = history
        fitted[name] = (model, gradcam_layer)
        labels, predictions, _ = collect_predictions(model, candidate_loaders["val"], device)
        validation_metrics[name] = compute_metrics(labels, predictions, config.num_classes)
        logger.info("[%s] validation f1_macro=%.3f", name, validation_metrics[name]["f1_macro"])

    if not fitted:
        logger.error("No candidate trained successfully")
        return 1

    best_name = max(validation_metrics, key=lambda n: validation_metrics[n][config.selection_metric])
    best_model, gradcam_layer = fitted[best_name]
    best_loaders = build_loaders(
        splits, config, config.batch_size_by_model.get(best_name, config.batch_size)
    )
    labels, predictions, probabilities = collect_predictions(best_model, best_loaders["test"], device)
    test_metrics = compute_metrics(labels, predictions, config.num_classes)

    dataset_info = {
        "source": "NASA GIBS MODIS Terra (True Color + Band31 IR), real scenes",
        "scenes": len(index),
        "date_range": (index["date"].min(), index["date"].max()),
        "class_counts": index["label"].value_counts().to_dict(),
        "splits": {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
    }
    selection_reason = (
        f"highest validation macro-F1 ({validation_metrics[best_name]['f1_macro']:.3f}) "
        "with chronological splits; accuracy not used (class imbalance)"
    )

    checkpoint_path = save_checkpoint(
        config=config, model_name=best_name, model=best_model,
        metrics={"validation": validation_metrics[best_name], "test": test_metrics},
        dataset_info=dataset_info, gradcam_target_layer=gradcam_layer,
    )

    curves = plot_training_curves(histories, config)
    confusion = plot_confusion_matrix(test_metrics, config)
    samples = plot_sample_predictions(splits.test, probabilities, predictions, config)
    features_path = extract_features_for_fusion(index, config)

    report_text = render_report(config, best_name, test_metrics, validation_metrics,
                                dataset_info, selection_reason)
    report_path = config.reports_dir / f"satellite_model_report_{config.model_version}.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")

    experiments = sorted(config.experiments_dir.glob("experiment_*.json"))
    experiment = {
        "id": f"experiment_{len(experiments) + 1:03d}",
        "date": pd.Timestamp.utcnow().isoformat(),
        "phase": "4-satellite",
        "model_version": config.model_version,
        "device": device,
        "candidates": {n: {"validation_metrics": {
            k: validation_metrics[n][k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        }, "epochs": len(histories[n]["train_loss"])} for n in validation_metrics},
        "selected_model": best_name,
        "selection_reason": selection_reason,
        "test_metrics": test_metrics,
        "dataset": dataset_info,
        "artifacts": {"checkpoint": str(checkpoint_path), "report": str(report_path),
                      "curves": curves, "confusion_matrix": confusion,
                      "sample_predictions": samples, "fusion_features": features_path},
    }
    (config.experiments_dir / f"{experiment['id']}.json").write_text(
        json.dumps(experiment, indent=2, default=str), encoding="utf-8"
    )

    print(report_text)
    print(f"\nCheckpoint: {checkpoint_path}\nReport:     {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

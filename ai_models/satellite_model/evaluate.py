"""
Satellite model evaluation: metrics, report, and visualizations.

Same metric discipline as Phase 3: macro-F1 for selection (accuracy is
reported with an imbalance caveat), per-class recall surfaced because a
missed heavy-rain scene is the costly error for an early-warning system.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

from ai_models.satellite_model.config import SatelliteConfig

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a loader → (labels, predictions, probabilities)."""
    model.eval()
    labels, predictions, probabilities = [], [], []
    for images, targets in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probabilities.append(probs)
        predictions.append(probs.argmax(axis=1))
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(predictions), np.concatenate(probabilities)


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> dict[str, Any]:
    class_ids = list(range(num_classes))
    per_precision = precision_score(labels, predictions, labels=class_ids, average=None, zero_division=0)
    per_recall = recall_score(labels, predictions, labels=class_ids, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class": {
            int(c): {
                "precision": float(per_precision[i]),
                "recall": float(per_recall[i]),
                "support": int((labels == c).sum()),
            }
            for i, c in enumerate(class_ids)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
        "labels": class_ids,
    }


def render_report(
    config: SatelliteConfig,
    best_name: str,
    test_metrics: dict[str, Any],
    comparison: dict[str, dict[str, Any]],
    dataset_info: dict[str, Any],
    selection_reason: str,
) -> str:
    verdict = "GOOD" if test_metrics["f1_macro"] >= config.good_f1_macro else "NEEDS IMPROVEMENT"
    lines = [
        "==============================================",
        "       VARUNA AI SATELLITE MODEL REPORT",
        "==============================================",
        f"Model:                {best_name}",
        f"Version:              {config.model_version}",
        "Task:                 Rainfall category from satellite cloud scene (0/1/2)",
        f"Imagery:              {dataset_info.get('scenes', '?')} MODIS scenes "
        f"({dataset_info.get('date_range', ('?', '?'))[0]} → {dataset_info.get('date_range', ('?', '?'))[1]})",
        "----------------------------------------------",
        f"Accuracy:             {test_metrics['accuracy'] * 100:.1f}%   (imbalance-inflated; not the selection metric)",
        f"Precision (macro):    {test_metrics['precision_macro'] * 100:.1f}%",
        f"Recall (macro):       {test_metrics['recall_macro'] * 100:.1f}%",
        f"F1 Score (macro):     {test_metrics['f1_macro'] * 100:.1f}%",
        "----------------------------------------------",
        "Per-class performance (held-out test period):",
    ]
    for class_id, stats in test_metrics["per_class"].items():
        name = config.label_names[int(class_id)]
        lines.append(
            f"  {name:<8} precision {stats['precision'] * 100:5.1f}%  "
            f"recall {stats['recall'] * 100:5.1f}%  (n={stats['support']})"
        )
    lines += ["----------------------------------------------",
              "Validation comparison (selection metric: macro-F1):"]
    for name, metrics in sorted(comparison.items(), key=lambda kv: -kv[1]["f1_macro"]):
        marker = " <- selected" if name == best_name else ""
        lines.append(
            f"  {name:<12} f1_macro {metrics['f1_macro']:.3f}  "
            f"recall_macro {metrics['recall_macro']:.3f}  accuracy {metrics['accuracy']:.3f}{marker}"
        )
    lines += [
        "----------------------------------------------",
        f"Selection reason:     {selection_reason}",
        f"Performance:          {verdict}",
        "==============================================",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Visualizations (reports/)
# ---------------------------------------------------------------------
def plot_training_curves(histories: dict[str, dict[str, list[float]]], config: SatelliteConfig) -> str:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        ax_loss.plot(epochs, history["train_loss"], label=f"{name} train")
        ax_loss.plot(epochs, history["val_loss"], linestyle="--", label=f"{name} val")
        ax_acc.plot(epochs, [a * 100 for a in history["val_accuracy"]], label=name)
    ax_loss.set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    ax_acc.set(title="Validation accuracy", xlabel="epoch", ylabel="%")
    ax_loss.legend(fontsize=8)
    ax_acc.legend(fontsize=8)
    fig.tight_layout()
    path = config.reports_dir / f"satellite_training_curves_{config.model_version}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(metrics: dict[str, Any], config: SatelliteConfig) -> str:
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    names = [config.label_names[c] for c in metrics["labels"]]
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Satellite model {config.model_version} — test confusion matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im)
    fig.tight_layout()
    path = config.reports_dir / f"satellite_confusion_matrix_{config.model_version}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_sample_predictions(
    frame, probabilities: np.ndarray, predictions: np.ndarray, config: SatelliteConfig, count: int = 8
) -> str:
    """Grid of test scenes with actual vs predicted labels."""
    from ai_models.satellite_model.config import REPO_ROOT
    from ai_models.satellite_model.preprocessing import load_rgb

    rows = frame.reset_index(drop=True)
    # Show a mix: prioritise positive-class scenes, fill with normals.
    order = np.argsort(-rows["label"].to_numpy())[:count]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for ax, idx in zip(axes.flat, order):
        row = rows.iloc[idx]
        image = load_rgb(REPO_ROOT / row["true_color_path"])
        actual = config.label_names[int(row["label"])]
        predicted = config.label_names[int(predictions[idx])]
        risk = float(probabilities[idx][1:].sum())
        ax.imshow(image)
        correct = actual == predicted
        ax.set_title(
            f"{row['region']} {row['date']}\nactual {actual} / pred {predicted} (risk {risk:.2f})",
            fontsize=8, color="green" if correct else "red",
        )
        ax.axis("off")
    fig.suptitle("Sample test predictions — satellite model")
    fig.tight_layout()
    path = config.reports_dir / f"satellite_sample_predictions_{config.model_version}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)

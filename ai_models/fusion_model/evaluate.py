"""
Fusion model evaluation: metrics, the hybrid report, and visualizations.

Metric discipline matches Phases 3 and 4 — macro-F1 selects, accuracy is
reported with an imbalance caveat, and per-class recall is surfaced
because a missed heavy-rain day is the costly error for an early-warning
system.

The report additionally quantifies what fusion actually bought: the best
hybrid is compared against each single-modality baseline on the same
held-out period, and the improvement is stated as a relative percentage
of macro-F1.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ai_models.fusion_model.config import FusionConfig

logger = logging.getLogger(__name__)


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> dict[str, Any]:
    """Standard metric block.

    Macro averages are restricted to classes that actually occur in the
    evaluation period (`present`), so a class with no support cannot drag
    the score to zero. The full per-class table is still reported for
    every class, including empty ones, so the gap stays visible.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    class_ids = list(range(num_classes))
    present = sorted(set(labels.tolist()) | set(predictions.tolist()))

    per_precision = precision_score(labels, predictions, labels=class_ids, average=None, zero_division=0)
    per_recall = recall_score(labels, predictions, labels=class_ids, average=None, zero_division=0)
    per_f1 = f1_score(labels, predictions, labels=class_ids, average=None, zero_division=0)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(labels, predictions, labels=present, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, labels=present, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels, predictions, labels=present, average="macro", zero_division=0)
        ),
        "per_class": {
            int(c): {
                "precision": float(per_precision[i]),
                "recall": float(per_recall[i]),
                "f1": float(per_f1[i]),
                "support": int((labels == c).sum()),
            }
            for i, c in enumerate(class_ids)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
        "evaluated_classes": present,
        "labels": class_ids,
    }


def improvement_percentage(fusion_score: float, baseline_score: float) -> float | None:
    """Relative improvement of the fusion over a baseline, in percent."""
    if baseline_score <= 0:
        return None
    return float((fusion_score - baseline_score) / baseline_score * 100.0)


def render_report(
    config: FusionConfig,
    best_name: str,
    best_approach: str,
    test_metrics: dict[str, Any],
    comparison: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    dataset_info: dict[str, Any],
    selection_reason: str,
    caveats: list[str],
) -> str:
    verdict = "GOOD" if test_metrics["f1_macro"] >= config.good_f1_macro else "NEEDS IMPROVEMENT"
    lines = [
        "==============================================",
        "        VARUNA AI HYBRID MODEL REPORT",
        "==============================================",
        f"Fusion model:         {best_name}  ({best_approach})",
        f"Version:              {config.model_version}",
        "Task:                 Next-day (T+1) region rainfall category from",
        "                      day-T weather + day-T satellite scene",
        f"Region-days:          {dataset_info.get('rows', '?')} "
        f"({dataset_info.get('date_range', ('?', '?'))[0]} → {dataset_info.get('date_range', ('?', '?'))[1]})",
        f"Regions:              {', '.join(dataset_info.get('regions', []))}",
        "----------------------------------------------",
        "INDIVIDUAL MODEL PERFORMANCE (same held-out test period)",
        "----------------------------------------------",
    ]
    fusion_label = f"FUSION ({best_name})"
    width = max([len(name) for name in baselines] + [len(fusion_label)])

    def row(label: str, metrics: dict[str, Any]) -> str:
        return (
            f"  {label:<{width}}  accuracy {metrics['accuracy'] * 100:5.1f}%   "
            f"macro-F1 {metrics['f1_macro'] * 100:5.1f}%   "
            f"macro-recall {metrics['recall_macro'] * 100:5.1f}%"
        )

    for name, metrics in baselines.items():
        lines.append(row(name, metrics))
    lines += [
        row(fusion_label, test_metrics),
        "----------------------------------------------",
        "IMPROVEMENT FROM FUSION (relative macro-F1)",
        "----------------------------------------------",
    ]
    for name, metrics in baselines.items():
        delta = improvement_percentage(test_metrics["f1_macro"], metrics["f1_macro"])
        rendered = f"{delta:+.1f}%" if delta is not None else "n/a (baseline scored 0)"
        lines.append(f"  vs {name:<16} {rendered}")

    lines += [
        "----------------------------------------------",
        "PER-CLASS PERFORMANCE (held-out test period)",
        "----------------------------------------------",
    ]
    for class_id, stats in test_metrics["per_class"].items():
        name = config.label_names[int(class_id)]
        note = "  [no test samples — not measurable]" if stats["support"] == 0 else ""
        lines.append(
            f"  {name:<8} precision {stats['precision'] * 100:5.1f}%  "
            f"recall {stats['recall'] * 100:5.1f}%  (n={stats['support']}){note}"
        )

    lines += [
        "----------------------------------------------",
        "APPROACH COMPARISON (validation macro-F1)",
        "----------------------------------------------",
    ]
    for name, metrics in sorted(comparison.items(), key=lambda kv: -kv[1]["f1_macro"]):
        marker = " <- selected" if name == best_name else ""
        lines.append(
            f"  {name:<24} f1_macro {metrics['f1_macro']:.3f}  "
            f"recall_macro {metrics['recall_macro']:.3f}  accuracy {metrics['accuracy']:.3f}{marker}"
        )

    lines += ["----------------------------------------------", f"Selection reason:     {selection_reason}"]
    if caveats:
        lines += ["----------------------------------------------", "CAVEATS"]
        lines += [f"  - {caveat}" for caveat in caveats]
    lines += [
        "----------------------------------------------",
        f"Performance:          {verdict}",
        "==============================================",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------
def plot_approach_comparison(comparison: dict[str, dict[str, Any]], config: FusionConfig) -> str:
    names = list(comparison)
    scores = [comparison[n]["f1_macro"] * 100 for n in names]
    order = np.argsort(scores)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 2.5))
    ax.barh([names[i] for i in order], [scores[i] for i in order], color="#2b6cb0")
    ax.set_xlabel("validation macro-F1 (%)")
    ax.set_title(f"VARUNA AI fusion approaches — {config.model_version}")
    for y, i in enumerate(order):
        ax.text(scores[i] + 0.5, y, f"{scores[i]:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1)
    fig.tight_layout()
    path = config.reports_dir / f"fusion_approach_comparison_{config.model_version}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(metrics: dict[str, Any], config: FusionConfig) -> str:
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    names = [config.label_names[c] for c in metrics["labels"]]
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="Predicted", ylabel="Actual",
           title=f"Fusion model {config.model_version} — test confusion matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    path = config.reports_dir / f"fusion_confusion_matrix_{config.model_version}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_weight_sweep(sweep: list[dict[str, float]], config: FusionConfig) -> str:
    weights = [entry["weather_weight"] for entry in sweep]
    scores = [entry["score"] * 100 for entry in sweep]
    best = int(np.argmax(scores))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(weights, scores, marker="o", markersize=3, color="#2b6cb0")
    ax.axvline(weights[best], linestyle="--", color="#c53030",
               label=f"best w={weights[best]:.2f} ({scores[best]:.1f}%)")
    ax.set(xlabel="weather weight (satellite gets 1 - w)",
           ylabel="validation macro-F1 (%)",
           title="Approach 1 — weighted fusion sweep")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = config.reports_dir / f"fusion_weight_sweep_{config.model_version}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)

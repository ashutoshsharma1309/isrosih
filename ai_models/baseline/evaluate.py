"""
Model evaluation: metrics, comparison, and the VARUNA AI MODEL REPORT.

Accuracy is reported but never used for selection — with ~99% of days
being "normal rainfall", a useless model that always predicts class 0
scores 99% accuracy. Selection uses macro-F1; the report additionally
surfaces per-class recall, which is what matters for an early-warning
system (a missed heavy-rain day is the costly error).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ai_models.baseline.config import BaselineConfig

logger = logging.getLogger(__name__)


def evaluate_model(estimator: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Compute the full metric set for one fitted model on one split."""
    predictions = estimator.predict(x)
    labels = sorted(np.unique(np.concatenate([np.unique(y), np.unique(predictions)])))
    per_class_recall = recall_score(y, predictions, labels=labels, average=None, zero_division=0)
    per_class_precision = precision_score(y, predictions, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_macro": float(precision_score(y, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, predictions, average="weighted", zero_division=0)),
        "per_class": {
            int(label): {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "support": int((y == label).sum()),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
        "labels": [int(label) for label in labels],
    }


def render_model_report(
    config: BaselineConfig,
    best_name: str,
    test_metrics: dict[str, Any],
    comparison: dict[str, dict[str, Any]],
    dataset_info: dict[str, Any],
) -> str:
    """The human-readable VARUNA AI MODEL REPORT."""
    verdict = "GOOD" if test_metrics["f1_macro"] >= config.good_f1_macro else "NEEDS IMPROVEMENT"
    lines = [
        "==============================================",
        "         VARUNA AI MODEL REPORT",
        "==============================================",
        f"Model:                {best_name}",
        f"Version:              {config.model_version}",
        f"Task:                 Next-day rainfall category (0 normal / 1 heavy / 2 extreme)",
        f"Training data:        {dataset_info.get('records', '?')} records, "
        f"{dataset_info.get('train_range', ('?', '?'))[0]} → {dataset_info.get('train_range', ('?', '?'))[1]}",
        f"Test period:          {dataset_info.get('test_range', ('?', '?'))[0]} → "
        f"{dataset_info.get('test_range', ('?', '?'))[1]} (held out, chronological)",
        "----------------------------------------------",
        f"Accuracy:             {test_metrics['accuracy'] * 100:.1f}%   (inflated by class imbalance — see below)",
        f"Precision (macro):    {test_metrics['precision_macro'] * 100:.1f}%",
        f"Recall (macro):       {test_metrics['recall_macro'] * 100:.1f}%",
        f"F1 Score (macro):     {test_metrics['f1_macro'] * 100:.1f}%",
        "----------------------------------------------",
        "Per-class performance (test set):",
    ]
    for label, stats in test_metrics["per_class"].items():
        name = config.label_names.get(int(label), str(label))
        lines.append(
            f"  {name:<18} precision {stats['precision'] * 100:5.1f}%  "
            f"recall {stats['recall'] * 100:5.1f}%  (n={stats['support']})"
        )
    lines += [
        "----------------------------------------------",
        "Confusion matrix (rows = actual, cols = predicted):",
    ]
    header = "        " + "".join(f"{config.label_names[l][:7]:>9}" for l in test_metrics["labels"])
    lines.append(header)
    for label, row in zip(test_metrics["labels"], test_metrics["confusion_matrix"]):
        lines.append(f"  {config.label_names[label][:7]:<7}" + "".join(f"{v:>9}" for v in row))
    lines += [
        "----------------------------------------------",
        "Validation-set comparison (selection metric: macro-F1):",
    ]
    for name, metrics in sorted(comparison.items(), key=lambda kv: -kv[1]["f1_macro"]):
        marker = " <- selected" if name == best_name else ""
        lines.append(
            f"  {name:<22} f1_macro {metrics['f1_macro']:.3f}  "
            f"recall_macro {metrics['recall_macro']:.3f}  accuracy {metrics['accuracy']:.3f}{marker}"
        )
    lines += [
        "----------------------------------------------",
        f"Performance:          {verdict}",
        "==============================================",
    ]
    return "\n".join(lines)

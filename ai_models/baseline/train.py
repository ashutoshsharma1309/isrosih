"""
Baseline model training pipeline.

    load (READY-gated) → engineer features → chronological split
    → train 4 candidates → select on validation macro-F1
    → refit winner on train+validation → single held-out test evaluation
    → save bundle + model report + feature importance + experiment record

Run (from the repository root, using the backend virtualenv):
    backend/.venv/bin/python -m ai_models.baseline.train
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering for report images
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

from ai_models.baseline import data as data_prep
from ai_models.baseline.config import BaselineConfig
from ai_models.baseline.evaluate import evaluate_model, render_model_report
from ai_models.baseline.model import build_candidate_models, needs_sample_weight, save_bundle

logger = logging.getLogger(__name__)


def fit_model(name: str, estimator: Any, x: pd.DataFrame, y: pd.Series) -> Any:
    """Fit one candidate, applying balanced sample weights where the
    estimator has no class_weight support of its own."""
    if needs_sample_weight(name):
        weights = compute_sample_weight("balanced", y)
        estimator.fit(x, y, sample_weight=weights)
    else:
        estimator.fit(x, y)
    return estimator


def compute_feature_importance(
    estimator: Any, x_test: pd.DataFrame, y_test: pd.Series, seed: int
) -> dict[str, float]:
    """Permutation importance (model-agnostic), normalized to shares.

    This is the pre-SHAP importance view; Phase 6 replaces it with
    per-prediction SHAP attributions.
    """
    result = permutation_importance(
        estimator, x_test, y_test, scoring="f1_macro", n_repeats=5, random_state=seed, n_jobs=-1
    )
    raw = {name: max(float(v), 0.0) for name, v in zip(x_test.columns, result.importances_mean)}
    total = sum(raw.values()) or 1.0
    return dict(sorted(((k, v / total) for k, v in raw.items()), key=lambda kv: -kv[1]))


def plot_feature_importance(importance: dict[str, float], config: BaselineConfig) -> str:
    fig, ax = plt.subplots(figsize=(9, 6))
    names = list(importance)[::-1]
    values = [importance[n] * 100 for n in names]
    ax.barh(names, values, color="#2563eb")
    ax.set_xlabel("Importance share (%) — permutation importance, macro-F1")
    ax.set_title(f"VARUNA AI baseline {config.model_version}: feature importance")
    fig.tight_layout()
    out_path = config.reports_dir / f"feature_importance_{config.model_version}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def next_experiment_id(config: BaselineConfig) -> str:
    existing = sorted(config.experiments_dir.glob("experiment_*.json"))
    return f"experiment_{len(existing) + 1:03d}"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    config = BaselineConfig()
    config.ensure_output_dirs()

    # --- Data -----------------------------------------------------------
    dataset = data_prep.load_dataset(config)
    frame = data_prep.build_supervised_frame(dataset, config)
    splits = data_prep.time_based_split(frame, config)

    # --- Train and compare candidates ------------------------------------
    candidates = build_candidate_models(config.random_seed)
    validation_results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Any] = {}
    for name, estimator in candidates.items():
        logger.info("Training %s ...", name)
        fitted[name] = fit_model(name, estimator, splits.x_train, splits.y_train)
        validation_results[name] = evaluate_model(fitted[name], splits.x_val, splits.y_val)
        logger.info(
            "%s: val f1_macro=%.3f recall_macro=%.3f",
            name, validation_results[name]["f1_macro"], validation_results[name]["recall_macro"],
        )

    best_name = max(validation_results, key=lambda n: validation_results[n][config.selection_metric])
    logger.info("Selected model: %s (by validation %s)", best_name, config.selection_metric)

    # --- Refit winner on train+val, single test evaluation ----------------
    x_trainval = pd.concat([splits.x_train, splits.x_val])
    y_trainval = pd.concat([splits.y_train, splits.y_val])
    final_estimator = fit_model(best_name, clone(candidates[best_name]), x_trainval, y_trainval)
    test_metrics = evaluate_model(final_estimator, splits.x_test, splits.y_test)

    dataset_info = {
        "source": "NASA POWER daily point data (real, via Phase 2 pipeline)",
        "dataset_path": str(config.dataset_path),
        "records": len(frame),
        "train_range": splits.train_range,
        "test_range": splits.test_range,
        "class_counts": frame[config.target_column].value_counts().to_dict(),
    }

    # --- Feature importance ----------------------------------------------
    importance = compute_feature_importance(
        final_estimator, splits.x_test, splits.y_test, config.random_seed
    )
    importance_plot = plot_feature_importance(importance, config)
    (config.reports_dir / f"feature_importance_{config.model_version}.json").write_text(
        json.dumps(importance, indent=2), encoding="utf-8"
    )

    # --- Persist artifacts -----------------------------------------------
    medians = {name: float(x_trainval[name].median()) for name in splits.feature_names}
    bundle_path = save_bundle(
        config=config,
        model_name=best_name,
        estimator=final_estimator,
        feature_names=splits.feature_names,
        feature_medians=medians,
        metrics={"validation": validation_results[best_name], "test": test_metrics},
        dataset_info=dataset_info,
    )

    report_text = render_model_report(config, best_name, test_metrics, validation_results, dataset_info)
    report_path = config.reports_dir / f"model_report_{config.model_version}.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")

    experiment = {
        "id": next_experiment_id(config),
        "date": datetime.now(timezone.utc).isoformat(),
        "model_version": config.model_version,
        "task": "next-day rainfall category (0/1/2)",
        "features": splits.feature_names,
        "dataset": dataset_info,
        "candidates": {
            name: {
                "params": {k: v for k, v in _safe_params(candidates[name]).items()},
                "validation_metrics": {
                    k: validation_results[name][k]
                    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
                },
            }
            for name in candidates
        },
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "feature_importance": importance,
        "artifacts": {"bundle": str(bundle_path), "report": str(report_path),
                      "importance_plot": importance_plot},
    }
    experiment_path = config.experiments_dir / f"{experiment['id']}.json"
    experiment_path.write_text(json.dumps(experiment, indent=2, default=str), encoding="utf-8")

    print(report_text)
    print(f"\nBundle:     {bundle_path}\nReport:     {report_path}\nExperiment: {experiment_path}")
    return 0


def _safe_params(estimator: Any) -> dict[str, Any]:
    """JSON-safe hyperparameters for the experiment record."""
    try:
        params = estimator.get_params(deep=False)
    except Exception:  # pragma: no cover
        return {}
    return {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool, type(None)))}


if __name__ == "__main__":
    sys.exit(main())

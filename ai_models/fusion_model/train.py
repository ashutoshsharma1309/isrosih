"""
Hybrid fusion training pipeline.

    build unified (weather + satellite) region-day dataset
    → chronological split
    → score single-modality baselines (what fusion must beat)
    → fit all three fusion approaches
    → select on validation macro-F1 → single held-out test evaluation
    → fit historical matcher → save bundle + model_info.json
    → VARUNA AI HYBRID MODEL REPORT + comparison/confusion/sweep plots

Run (from the repository root):
    backend/.venv/bin/python -m ai_models.fusion_model.train
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ai_models.fusion_model.config import (
    FUSION_BUNDLE_TEMPLATE,
    MODEL_INFO_FILENAME,
    FusionConfig,
)
from ai_models.fusion_model.dataset import build_unified_dataset, chronological_split
from ai_models.fusion_model.evaluate import (
    compute_metrics,
    improvement_percentage,
    plot_approach_comparison,
    plot_confusion_matrix,
    plot_weight_sweep,
    render_report,
)
from ai_models.fusion_model.fusion import (
    FeatureFusion,
    NeuralFusion,
    SingleModality,
    WeightedFusion,
    fit_weight,
)
from ai_models.fusion_model.historical_match import MATCH_FEATURES, HistoricalMatcher
from ai_models.fusion_model.utils import configure_logging, set_seeds

logger = logging.getLogger(__name__)


def _macro_f1(y_true, y_pred) -> float:
    """Macro-F1 over classes present in the split (see evaluate.compute_metrics)."""
    present = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def _evaluate(strategy, frame: pd.DataFrame, target: pd.Series, config: FusionConfig) -> dict[str, Any]:
    predictions = strategy.predict_proba(frame).argmax(axis=1)
    return compute_metrics(target.to_numpy(), predictions, config.num_classes)


def _split_auc(column: str, splits, config: FusionConfig) -> dict[str, float]:
    """Per-split AUC of one feature against the binary high-impact target.

    A feature that scores far higher on the split its source model was
    fitted on than on later data is memorised, not predictive — the check
    that caught the Phase 3 leak.
    """
    from sklearn.metrics import roc_auc_score

    parts = {
        "train": (splits.train_frame, splits.y_train),
        "validation": (splits.val_frame, splits.y_val),
        "test": (splits.test_frame, splits.y_test),
    }
    scores: dict[str, float] = {}
    for name, (frame, target) in parts.items():
        binary = (target > 0).astype(int)
        if binary.nunique() < 2 or column not in frame:
            continue
        scores[name] = float(roc_auc_score(binary, frame[column]))
    return scores


def build_caveats(
    unified: pd.DataFrame,
    splits,
    config: FusionConfig,
    test_metrics: dict[str, Any],
    baselines_test: dict[str, dict[str, Any]],
    best: Any,
) -> list[str]:
    """State the limitations the headline numbers would otherwise hide."""
    caveats: list[str] = []

    counts = unified[config.target_column].value_counts().to_dict()
    extreme = int(counts.get(2, 0))
    if int((splits.y_test == 2).sum()) == 0:
        caveats.append(
            f"Extreme (>=204.4 mm) is not measurable: {extreme} such region-day(s) survive into "
            "the fusion dataset and none fall in the held-out test period. Every Extreme figure "
            "in this report is structural, not evidence the class works. Needs IMD gridded / "
            "GPM IMERG data (already flagged for Phases 3 and 4)."
        )

    positives = int((splits.y_test > 0).sum())
    caveats.append(
        f"The comparison is underpowered: selection rests on {len(splits.y_val)} validation "
        f"region-days and the verdict on {len(splits.y_test)} test region-days holding only "
        f"{positives} high-impact day(s). Differences of a few macro-F1 points here are noise, "
        "not evidence one approach is better."
    )

    if isinstance(best, WeightedFusion) and best.weather_weight in (0.0, 1.0):
        kept, dropped = (
            ("weather", "satellite") if best.weather_weight == 1.0 else ("satellite", "weather")
        )
        caveats.append(
            f"The selected blend is degenerate: the validation sweep chose weight "
            f"{best.weather_weight:.2f}, which discards the {dropped} branch entirely, so this "
            f"'fusion' is the {kept} model unchanged. On this data no blend of the two branches "
            "beat the stronger branch alone."
        )

    best_baseline = max(baselines_test.items(), key=lambda kv: kv[1]["f1_macro"])
    if test_metrics["f1_macro"] < best_baseline[1]["f1_macro"]:
        caveats.append(
            f"Fusion did NOT beat the single-modality baseline on the held-out test period: "
            f"{best_baseline[0]} scored macro-F1 {best_baseline[1]['f1_macro']:.3f} against the "
            f"fusion's {test_metrics['f1_macro']:.3f}. The fusion wins on validation and loses on "
            "test, which is the signature of selection overfitting on a split this small."
        )
    heavy = test_metrics["per_class"].get(1, {})
    if heavy.get("support", 0) and heavy.get("recall", 1.0) < best_baseline[1]["recall_macro"]:
        caveats.append(
            f"The selected fusion is conservative in the wrong direction: it recalls only "
            f"{heavy['recall'] * 100:.0f}% of Heavy days while raising overall accuracy. For an "
            "early-warning system a missed event costs more than a false alarm, so macro-F1 "
            "selection should be revisited against a recall- or cost-weighted criterion."
        )

    weather_auc = _split_auc("weather_risk_score", splits, config)
    rendered = " / ".join(f"{value:.3f} {name}" for name, value in weather_auc.items())
    caveats.append(
        "weather_risk_score is generated out-of-fold by walk-forward retraining "
        f"({config.oof_folds} folds): every region-day is scored by a Phase 3 estimator fitted "
        f"only on strictly earlier days, giving AUC {rendered}. Scoring with the shipped Phase 3 "
        "artifact instead measured AUC 1.000 on train and validation against 0.875 on test — "
        "memorisation, which made all trained approaches tie at a meaningless validation "
        "macro-F1 of 1.000."
    )
    auc = _split_auc("satellite_risk_score", splits, config)
    if auc:
        rendered = " / ".join(f"{value:.3f} {name}" for name, value in auc.items())
        caveats.append(
            "satellite_risk_score is left in-sample: the Phase 4 checkpoint saw the fusion "
            f"training scenes, but its measured AUC is stable across splits ({rendered}), so it "
            "shows no comparable memorisation. Walk-forward retraining of the CNN was not run "
            "for this phase."
        )
    caveats.append(
        "The satellite branch classifies day T while the target is day T+1, so it contributes "
        "current convective state rather than a forecast; its standalone score on this task is "
        "correspondingly weaker than its Phase 4 nowcast score."
    )
    caveats.append(
        f"Fusion is limited to the {len(unified)} region-days that have both modalities and an "
        "out-of-fold weather score — satellite coverage, not weather coverage, is the binding "
        "constraint."
    )
    return caveats


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    config = FusionConfig()
    config.ensure_output_dirs()
    set_seeds(config.random_seed)

    # --- Data ---------------------------------------------------------
    unified = build_unified_dataset(config)
    splits = chronological_split(unified, config)

    # --- Single-modality baselines (the bar fusion must clear) --------
    baselines_val: dict[str, dict[str, Any]] = {}
    baselines_test: dict[str, dict[str, Any]] = {}
    for modality in ("weather", "satellite"):
        baseline = SingleModality(modality).fit(splits.x_train, splits.y_train, config)
        baselines_val[baseline.name] = _evaluate(baseline, splits.val_frame, splits.y_val, config)
        baselines_test[baseline.name] = _evaluate(baseline, splits.test_frame, splits.y_test, config)
        logger.info(
            "[%s] validation f1_macro=%.3f  test f1_macro=%.3f",
            baseline.name, baselines_val[baseline.name]["f1_macro"],
            baselines_test[baseline.name]["f1_macro"],
        )

    # --- Approach 1: weighted late fusion ------------------------------
    weighted, sweep = fit_weight(splits.val_frame, splits.y_val, config, _macro_f1)

    # --- Approaches 2 and 3: trained fusions ---------------------------
    strategies: list[Any] = [weighted]
    for estimator_name in ("random_forest", "xgboost"):
        try:
            strategies.append(
                FeatureFusion(estimator_name, config.random_seed).fit(
                    splits.x_train, splits.y_train, config
                )
            )
        except Exception as exc:  # a missing optional dependency must not sink the phase
            logger.warning("Feature fusion %s unavailable (%s) — skipping", estimator_name, exc)
    try:
        strategies.append(NeuralFusion(config.random_seed).fit(splits.x_train, splits.y_train, config))
    except Exception as exc:
        logger.warning("Neural fusion unavailable (%s) — skipping", exc)

    # --- Select on validation ------------------------------------------
    comparison: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        comparison[strategy.name] = _evaluate(strategy, splits.val_frame, splits.y_val, config)
        logger.info("[%s] validation f1_macro=%.3f", strategy.name, comparison[strategy.name]["f1_macro"])
    comparison.update({f"{name} (baseline)": metrics for name, metrics in baselines_val.items()})

    trained_names = {strategy.name for strategy in strategies}
    best_name = max(
        (name for name in comparison if name in trained_names),
        key=lambda name: comparison[name][config.selection_metric],
    )
    best = next(strategy for strategy in strategies if strategy.name == best_name)
    test_metrics = _evaluate(best, splits.test_frame, splits.y_test, config)
    logger.info("Selected %s (%s) — test f1_macro=%.3f", best.name, best.approach, test_metrics["f1_macro"])

    # --- Historical analogues ------------------------------------------
    reference = pd.concat([splits.train_frame, splits.val_frame])
    matcher = HistoricalMatcher().fit(reference, config)

    # --- Persist --------------------------------------------------------
    dataset_info = {
        "rows": len(unified),
        "regions": sorted(unified["region"].unique().tolist()),
        "date_range": (str(unified["date"].min()), str(unified["date"].max())),
        "target_counts": {int(k): int(v) for k, v in unified[config.target_column].value_counts().items()},
        "splits": {"train": len(splits.x_train), "val": len(splits.x_val), "test": len(splits.x_test)},
        "source": "Phase 2 NASA POWER weather + Phase 4 NASA GIBS MODIS scene features",
    }
    selection_reason = (
        f"highest validation macro-F1 ({comparison[best_name]['f1_macro']:.3f}) among the three "
        "fusion approaches, chronological splits, accuracy not used (class imbalance)"
    )
    caveats = build_caveats(unified, splits, config, test_metrics, baselines_test, best)

    bundle = {
        "version": config.model_version,
        "model_name": best.name,
        "approach": best.approach,
        "strategy": best,
        "matcher": matcher,
        "feature_names": config.feature_columns(),
        "match_features": list(MATCH_FEATURES),
        "label_names": config.label_names,
        "event_names": config.event_names,
        "feature_medians": splits.x_train.median(numeric_only=True).to_dict(),
        "metrics": {"validation": comparison[best_name], "test": test_metrics},
        "dataset_info": dataset_info,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
    }
    bundle_path = config.saved_models_dir / FUSION_BUNDLE_TEMPLATE.format(version=config.model_version)
    joblib.dump(bundle, bundle_path)
    logger.info("Saved fusion bundle: %s (%s)", bundle_path, best.name)

    improvements = {
        name: improvement_percentage(test_metrics["f1_macro"], metrics["f1_macro"])
        for name, metrics in baselines_test.items()
    }
    model_info = {
        "model_version": config.model_version,
        "model_name": best.name,
        "approach": best.approach,
        "training_date": bundle["trained_at"],
        "features_used": {
            "fusion_vector": config.feature_columns(),
            "historical_match": list(MATCH_FEATURES),
        },
        "performance_metrics": {
            "validation": comparison[best_name],
            "test": test_metrics,
            "single_modality_test": baselines_test,
            "improvement_pct_macro_f1": improvements,
        },
        "dataset": dataset_info,
        "upstream_models": {
            "weather": "ai_models/saved_models/rainfall_model_v1.pkl (Phase 3)",
            "satellite": "ai_models/saved_models/satellite_model_v1.pt (Phase 4)",
        },
        "caveats": caveats,
    }
    info_path = config.saved_models_dir / MODEL_INFO_FILENAME
    info_path.write_text(json.dumps(model_info, indent=2, default=str), encoding="utf-8")

    # --- Report + plots --------------------------------------------------
    report_text = render_report(
        config, best.name, best.approach, test_metrics, comparison,
        baselines_test, dataset_info, selection_reason, caveats,
    )
    report_path = config.reports_dir / f"hybrid_model_report_{config.model_version}.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")

    artifacts = {
        "bundle": str(bundle_path),
        "model_info": str(info_path),
        "report": str(report_path),
        "approach_comparison": plot_approach_comparison(comparison, config),
        "confusion_matrix": plot_confusion_matrix(test_metrics, config),
        "weight_sweep": plot_weight_sweep(sweep, config),
        "unified_dataset": str(config.unified_dataset_path),
    }

    experiments = sorted(config.experiments_dir.glob("experiment_*.json"))
    experiment = {
        "id": f"experiment_{len(experiments) + 1:03d}",
        "date": pd.Timestamp.now("UTC").isoformat(),
        "phase": "5-fusion",
        "model_version": config.model_version,
        "approaches": {
            name: {
                k: comparison[name][k]
                for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
            }
            for name in comparison
        },
        "weight_sweep": sweep,
        "selected_model": best.name,
        "selected_approach": best.approach,
        "selection_reason": selection_reason,
        "test_metrics": test_metrics,
        "single_modality_test": baselines_test,
        "improvement_pct_macro_f1": improvements,
        "dataset": dataset_info,
        "caveats": caveats,
        "artifacts": artifacts,
    }
    (config.experiments_dir / f"{experiment['id']}.json").write_text(
        json.dumps(experiment, indent=2, default=str), encoding="utf-8"
    )

    print(report_text)
    print(f"\nBundle:     {bundle_path}\nModel info: {info_path}\nReport:     {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

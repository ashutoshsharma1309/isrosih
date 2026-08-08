"""
XAI report generation.

Runs the explanation pipeline over a set of region-days and persists the
results: a machine-readable JSON payload, a human-readable text report,
the per-prediction SHAP and Grad-CAM figures, and the model-wide SHAP
summary plot.

Run (from the repository root):
    backend/.venv/bin/python -m explainability.reports.writer
    backend/.venv/bin/python -m explainability.reports.writer --region Kerala --date 2018-08-15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from explainability.explanation_generator.generator import ExplanationGenerator, UnifiedExplanation
from explainability.gradcam_explainer.config import GradCamConfig
from explainability.shap_explainer.config import ShapConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SATELLITE_INDEX = REPO_ROOT / "data/labels/satellite_labels.parquet"


def scene_for(region: str, date: str) -> Path | None:
    """Locate the True Color scene for a region-day, if one was acquired."""
    if not SATELLITE_INDEX.exists():
        logger.warning("Satellite index missing at %s", SATELLITE_INDEX)
        return None
    index = pd.read_parquet(SATELLITE_INDEX)
    match = index[(index["region"].astype(str) == region) & (index["date"].astype(str) == date)]
    if match.empty:
        return None
    path = REPO_ROOT / str(match.iloc[0]["true_color_path"])
    return path if path.exists() else None


def select_cases(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Pick region-days worth explaining.

    Chooses observed high-impact days first — those are the ones an
    operator would interrogate — then fills with the highest-risk
    remaining days so the report also shows a confident negative.
    """
    positives = frame[frame["target_category"] > 0].sort_values("weather_risk_score", ascending=False)
    negatives = frame[frame["target_category"] == 0].sort_values("weather_risk_score", ascending=False)
    chosen = pd.concat([positives.head(max(count - 1, 1)), negatives.head(1)])
    return chosen.head(count)


def render_text_report(explanations: list[UnifiedExplanation], bundle: dict[str, Any]) -> str:
    """The VARUNA AI EXPLANATION REPORT."""
    lines = [
        "==============================================",
        "        VARUNA AI EXPLANATION REPORT",
        "==============================================",
        f"Fusion model:         {bundle.get('model_name')}  ({bundle.get('approach')})",
        f"Explained cases:      {len(explanations)}",
        "SHAP:                 exact TreeSHAP via the branch that reads each input",
        "Grad-CAM:             Captum LayerGradCam on the Phase 4 checkpoint",
        "==============================================",
    ]
    for explanation in explanations:
        context = explanation.context
        lines += [
            "",
            f"--- {context.get('region')} {context.get('date')} ---",
            f"Prediction:   {explanation.prediction}  "
            f"({explanation.probability * 100:.1f}% probability, {explanation.risk_level})",
            f"Confidence:   {explanation.confidence.label} "
            f"({explanation.confidence.confidence_pct:.0f}%)  "
            f"[agreement "
            f"{_fmt(explanation.confidence.model_agreement)}, similarity "
            f"{_fmt(explanation.confidence.historical_similarity)}, data "
            f"{explanation.confidence.data_quality}]",
            "Top contributing factors:",
        ]
        for rank, item in enumerate(explanation.shap.top(5), start=1):
            flag = "  [median-filled]" if item.imputed else ""
            lines.append(
                f"  {rank}. {item.label:<24} {item.contribution:+.4f}  "
                f"{item.share * 100:+5.1f}%  {item.impact}{flag}"
            )
        if explanation.gradcam:
            regions = ", ".join(r.name for r in explanation.gradcam.regions) or "none isolated"
            lines.append(
                f"Satellite:    risk {explanation.gradcam.satellite_risk:.2f}, "
                f"{explanation.gradcam.coverage_label} coverage, regions: {regions}"
            )
        lines += ["Explanation:", f"  {explanation.narrative}"]

    caveats = _dedupe([c for e in explanations for c in e.caveats])
    if caveats:
        lines += ["", "----------------------------------------------", "CAVEATS"]
        lines += [f"  - {c}" for c in caveats]
    lines += ["==============================================" ]
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate VARUNA AI explanation reports")
    parser.add_argument("--region", help="Explain a single region (with --date)")
    parser.add_argument("--date", help="Explain a single date, YYYY-MM-DD")
    parser.add_argument("--cases", type=int, default=3, help="How many region-days to explain")
    parser.add_argument("--no-summary", action="store_true", help="Skip the model-wide SHAP summary")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    shap_config = ShapConfig()
    gradcam_config = GradCamConfig()
    shap_config.ensure_output_dirs()
    gradcam_config.ensure_output_dirs()

    if not shap_config.fusion_dataset_path.exists():
        print(
            f"No fusion dataset at {shap_config.fusion_dataset_path}. "
            "Run `python -m ai_models.fusion_model.train` first.",
            file=sys.stderr,
        )
        return 1

    frame = pd.read_parquet(shap_config.fusion_dataset_path)
    if args.region and args.date:
        cases = frame[
            (frame["region"].astype(str) == args.region)
            & (frame["date"].astype(str) == args.date)
        ]
        if cases.empty:
            print(f"No region-day {args.region} {args.date} in the fusion dataset.", file=sys.stderr)
            return 1
    else:
        cases = select_cases(frame, args.cases)

    generator = ExplanationGenerator(shap_config=shap_config, gradcam_config=gradcam_config)
    explanations: list[UnifiedExplanation] = []
    for index in range(len(cases)):
        row = cases.iloc[[index]]
        region, date = str(row.iloc[0]["region"]), str(row.iloc[0]["date"])
        logger.info("Explaining %s %s", region, date)
        explanations.append(
            generator.explain(
                row,
                image_path=scene_for(region, date),
                render=True,
                exclude_self=(region, date),
            )
        )

    payload = [e.to_dict() for e in explanations]
    json_path = shap_config.reports_dir.parent / "xai_explanations_v1.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    report_text = render_text_report(explanations, generator.bundle)
    report_path = shap_config.reports_dir.parent / "xai_report_v1.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")

    if not args.no_summary:
        from explainability.shap_explainer.visualization import plot_summary

        logger.info("Rendering model-wide SHAP summary (this scores many rows)")
        plot_summary(generator.shap_explainer, frame, shap_config)

    print(report_text)
    print(f"\nJSON:   {json_path}\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

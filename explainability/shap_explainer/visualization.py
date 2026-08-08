"""
SHAP visualizations for the hybrid model.

Three figures, all rendered from measured attributions:

- **bar** — ranked feature importance for one prediction
- **waterfall** — how the baseline risk becomes this prediction, step by step
- **summary** — overall model behaviour across a sample of the record

Written to `reports/shap/`. Colour is used consistently throughout: red
raises risk, blue lowers it. Rendering uses the Agg backend so the module
is safe to import from a headless process.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explainability.shap_explainer.config import ShapConfig
from explainability.shap_explainer.explainer import FusionShapExplainer, ShapExplanation

logger = logging.getLogger(__name__)

RAISES = "#c0392b"  # red — pushes risk up
LOWERS = "#2471a3"  # blue — pushes risk down


def _colour(value: float) -> str:
    return RAISES if value > 0 else LOWERS


def plot_feature_importance(
    explanation: ShapExplanation, config: ShapConfig, *, filename: str = "shap_bar.png"
) -> Path:
    """Ranked bar chart of the features driving one prediction."""
    config.ensure_output_dirs()
    items = explanation.top(config.top_k)[::-1]
    if not items:
        raise ValueError("No attributions to plot")

    labels = [f"{c.label}\n({c.value:g})" if c.value is not None else c.label for c in items]
    values = [c.contribution for c in items]

    fig, ax = plt.subplots(figsize=(9, 0.62 * len(items) + 2.2))
    ax.barh(labels, values, color=[_colour(v) for v in values])
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("contribution to fused risk (probability units)")
    ax.set_title("Why this risk score — ranked feature contributions")
    span = max(abs(v) for v in values) or 1.0
    for y, item in enumerate(items):
        offset = span * 0.02
        ax.text(
            item.contribution + (offset if item.contribution >= 0 else -offset),
            y,
            f"{item.share * 100:+.1f}%  {item.impact}",
            va="center",
            ha="left" if item.contribution >= 0 else "right",
            fontsize=8,
        )
    ax.set_xlim(-span * 1.45, span * 1.45)
    fig.tight_layout()
    path = config.reports_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("SHAP bar chart -> %s", path)
    return path


def plot_waterfall(
    explanation: ShapExplanation, config: ShapConfig, *, filename: str = "shap_waterfall.png"
) -> Path:
    """Baseline → prediction, one cumulative step per feature.

    Features outside the top-k are pooled into a single "other features"
    step so the bars still sum to the explained risk rather than silently
    dropping mass.
    """
    config.ensure_output_dirs()
    top = explanation.top(config.top_k)
    remainder = sum(
        c.contribution for c in explanation.contributions if c not in top
    )

    steps = [(c.label, c.contribution) for c in top]
    if abs(remainder) > 1e-9:
        steps.append((f"other features ({len(explanation.contributions) - len(top)})", remainder))

    fig, ax = plt.subplots(figsize=(10, 0.62 * len(steps) + 3.0))
    cumulative = explanation.base_value
    for index, (label, value) in enumerate(steps):
        ax.barh(index, value, left=cumulative, color=_colour(value), height=0.62)
        ax.text(
            cumulative + value / 2, index, f"{value:+.3f}",
            ha="center", va="center", fontsize=8,
            color="white" if abs(value) > 0.02 else "#222",
        )
        cumulative += value

    ax.set_yticks(range(len(steps)), [label for label, _ in steps])
    ax.invert_yaxis()
    ax.axvline(explanation.base_value, color="#555", linestyle="--", linewidth=1,
               label=f"baseline {explanation.base_value:.3f}")
    ax.axvline(explanation.predicted_risk, color="#111", linewidth=1.2,
               label=f"explained risk {explanation.predicted_risk:.3f}")
    ax.set_xlabel("fused risk (probability units)")
    ax.set_title("From baseline risk to this prediction")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = config.reports_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("SHAP waterfall -> %s", path)
    return path


def plot_summary(
    explainer: FusionShapExplainer,
    frame: pd.DataFrame,
    config: ShapConfig,
    *,
    filename: str = "shap_summary.png",
    sample_size: int | None = None,
) -> Path:
    """Overall model behaviour: attribution spread across many region-days.

    Each row is a feature; each dot is one region-day, positioned by its
    attribution and coloured by whether that day's feature value was high
    or low. Reading it: a feature whose high values sit on the right raises
    risk when it rises.
    """
    config.ensure_output_dirs()
    size = sample_size or config.summary_sample_size
    sample = frame.sample(min(size, len(frame)), random_state=0) if len(frame) > size else frame

    rows: list[dict[str, float]] = []
    values: list[dict[str, float]] = []
    for index in range(len(sample)):
        explanation = explainer.explain(sample.iloc[[index]])
        rows.append({c.feature: c.contribution for c in explanation.contributions})
        values.append({c.feature: (c.value if c.value is not None else np.nan)
                       for c in explanation.contributions})

    attributions = pd.DataFrame(rows).fillna(0.0)
    observed = pd.DataFrame(values)
    ranking = attributions.abs().mean().sort_values(ascending=False)
    features = list(ranking.index[: config.top_k])[::-1]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(features) + 2.6))
    rng = np.random.default_rng(0)
    for y, feature in enumerate(features):
        x = attributions[feature].to_numpy()
        raw = observed[feature].to_numpy(dtype=float)
        finite = np.isfinite(raw)
        if finite.sum() > 1 and np.nanmax(raw[finite]) > np.nanmin(raw[finite]):
            low, high = np.nanmin(raw[finite]), np.nanmax(raw[finite])
            shade = np.clip((raw - low) / (high - low), 0, 1)
        else:
            shade = np.full_like(x, 0.5, dtype=float)
        jitter = rng.uniform(-0.16, 0.16, size=len(x))
        scatter = ax.scatter(x, y + jitter, c=shade, cmap="coolwarm", s=13,
                             alpha=0.75, linewidths=0)

    ax.set_yticks(range(len(features)), [config.label_for(f) for f in features])
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("contribution to fused risk (probability units)")
    ax.set_title(f"Overall model behaviour — {len(sample)} region-days")
    bar = fig.colorbar(scatter, ax=ax, pad=0.02)
    bar.set_label("feature value (low → high)", fontsize=8)
    fig.tight_layout()
    path = config.reports_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("SHAP summary (%d rows) -> %s", len(sample), path)
    return path

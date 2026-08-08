"""
Natural-language templates for VARUNA AI explanations.

Every sentence is assembled from measured values passed in by the
generator — the templates decide *phrasing*, never *content*. There is no
stored claim here that humidity or cloud density matters; if a feature is
named in a sentence it is because SHAP measured it into the top ranks for
that specific prediction.

Phrasing rules:
- A driver that lowers risk is described as lowering it, never dropped to
  make the narrative sound more decisive.
- Pressure is inverted: falling pressure raises risk, so a negative-value
  attribution is phrased as a *drop*.
- When a branch carries zero weight, the narrative says so rather than
  implying the system weighed evidence it ignored.
"""

from __future__ import annotations

from typing import Any, Sequence

#: Risk-level wording used in the operator-facing summary line. The
#: phrasing deliberately says "high-impact rainfall" rather than naming a
#: category, so the sentence cannot contradict the predicted event.
RISK_SENTENCE = {
    "LOW": "Conditions do not indicate a high-impact rainfall event",
    "MODERATE": "Conditions carry a moderate chance of high-impact rainfall",
    "HIGH": "Conditions indicate a high chance of high-impact rainfall",
    "CRITICAL": "Conditions indicate a critical risk of high-impact rainfall",
}

#: How a feature's movement is described when it pushes risk up or down.
DIRECTION_PHRASE = {
    ("humidity_pct", True): "elevated humidity",
    ("humidity_pct", False): "low humidity",
    ("pressure_hpa", True): "falling atmospheric pressure",
    ("pressure_hpa", False): "steady atmospheric pressure",
    ("temperature_c", True): "elevated temperature",
    ("temperature_c", False): "cool temperatures",
    ("wind_speed_ms", True): "strengthening winds",
    ("wind_speed_ms", False): "light winds",
    ("cloud_cover_pct", True): "extensive cloud cover",
    ("cloud_cover_pct", False): "limited cloud cover",
    ("rain_sum_1d", True): "heavy rainfall in the past 24 hours",
    ("rain_sum_1d", False): "little rainfall in the past 24 hours",
    ("rain_sum_3d", True): "sustained rainfall over the past three days",
    ("rain_sum_3d", False): "a dry preceding three days",
    ("rain_sum_7d", True): "a wet preceding week",
    ("rain_sum_7d", False): "a dry preceding week",
    ("rain_sum_30d", True): "a saturated preceding month",
    ("rain_sum_30d", False): "a dry preceding month",
    ("rain_trend_3d", True): "an intensifying rainfall trend",
    ("rain_trend_3d", False): "an easing rainfall trend",
    ("cloud_density", True): "dense cloud formation over the region",
    ("cloud_density", False): "sparse cloud cover over the region",
    ("cold_top_fraction", True): "cold convective cloud tops",
    ("cold_top_fraction", False): "warm cloud tops",
    ("spatial_dispersion", True): "an organised cloud system",
    ("spatial_dispersion", False): "scattered, disorganised cloud",
    ("cloud_growth_rate", True): "rapidly growing cloud cover",
    ("cloud_growth_rate", False): "shrinking cloud cover",
    ("satellite_risk_score", True): "the satellite model's own high reading",
    ("satellite_risk_score", False): "the satellite model's low reading",
}


def describe_driver(feature: str, label: str, raises_risk: bool) -> str:
    """Phrase one driver in plain language, falling back to its label."""
    phrase = DIRECTION_PHRASE.get((feature, raises_risk))
    if phrase:
        return phrase
    qualifier = "elevated" if raises_risk else "reduced"
    return f"{qualifier} {label.lower()}"


def join_clauses(clauses: Sequence[str]) -> str:
    """Join phrases into readable prose ('a, b and c')."""
    items = [c for c in clauses if c]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def format_probability(probability: float) -> str:
    """Render a probability without rounding a near-certainty to a certainty."""
    percentage = probability * 100
    if 99.0 <= percentage < 100.0:
        return f"{percentage:.1f}%"
    if 0.0 < percentage <= 1.0:
        return f"{percentage:.1f}%"
    return f"{percentage:.0f}%"


def headline(event: str, probability: float, risk_level: str) -> str:
    """The operator-facing first sentence."""
    stem = RISK_SENTENCE.get(risk_level.upper(), "Conditions were assessed")
    return f"{stem} ({event}, {format_probability(probability)} probability)."


def drivers_sentence(raising: Sequence[str], lowering: Sequence[str]) -> str:
    """One sentence naming what pushed risk up and what held it down."""
    if raising and lowering:
        return (
            f"Risk was raised mainly by {join_clauses(raising)}, while "
            f"{join_clauses(lowering)} pulled it back."
        )
    if raising:
        return f"Risk was raised mainly by {join_clauses(raising)}."
    if lowering:
        return f"Risk was held down mainly by {join_clauses(lowering)}."
    return ""


def satellite_sentence(
    regions: Sequence[dict[str, Any]], coverage_label: str, weight: float, risk: float
) -> str:
    """Describe what the imagery contributed — including when it is nothing."""
    if weight <= 1e-9:
        return (
            f"The satellite model read this scene as {risk * 100:.0f}% risk, but the fusion "
            "gives its branch zero weight, so it did not influence this forecast."
        )
    if not regions:
        return (
            "The satellite branch contributed, but no single area of the scene stood out as "
            "driving the classification."
        )
    named = regions[0]
    return (
        f"In the imagery the model focused on {named['position']} "
        f"({named['name']}), with {coverage_label} high-influence coverage."
    )


def confidence_sentence(
    confidence_pct: float, agreement: float | None, similarity: float | None, quality: str
) -> str:
    """Explain the confidence number from its measured components."""
    parts = [f"Confidence is {confidence_pct:.0f}%"]
    if agreement is not None:
        descriptor = "high" if agreement >= 0.75 else "moderate" if agreement >= 0.5 else "low"
        parts.append(f"{descriptor} agreement between the weather and satellite branches")
    if similarity is not None:
        parts.append(f"{similarity * 100:.0f}% similarity to the closest past event")
    parts.append(f"{quality.lower()} input data quality")
    return f"{parts[0]}, based on {join_clauses(parts[1:])}."


def historical_sentence(match: dict[str, Any] | None) -> str:
    """Name the closest analogue, or say plainly that there is none."""
    if not match:
        return "No comparable event was found in the historical record."
    return (
        f"These conditions most closely resemble {match['event']}, "
        f"at {match['similarity_pct']:.0f}% similarity."
    )

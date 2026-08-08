# VARUNA AI — Explanation Examples

Real output from `explainability/`, produced by
`python -m explainability.reports.writer` against the shipped v1 artifacts.
Nothing here is illustrative or hand-written: every number below is a
measured SHAP value, a Captum attribution, or a stored model output. The
full payloads are in `reports/xai_explanations_v1.json`.

Read alongside `docs/explainable_ai_architecture.md`.

---

## Example 1 — Mumbai, 17 July 2017

The clearest case: heavy monsoon rainfall with both branches in strong
agreement and a near-identical event the following day in the record.

**Unified output**

```json
{
  "prediction": "Heavy Rainfall",
  "probability": 0.9973,
  "confidence": "High",
  "confidence_pct": 95.4,
  "risk_level": "CRITICAL",
  "top_features": ["Rainfall, past 24h", "Rainfall, past 3 days", "Rainfall, past 7 days"],
  "satellite_regions": ["cloud_cluster_area_1", "cloud_cluster_area_2"],
  "explanation": "Conditions indicate a critical risk of high-impact rainfall (Heavy Rainfall, 99.7% probability). Risk was raised mainly by heavy rainfall in the past 24 hours, sustained rainfall over the past three days and a wet preceding week. The satellite model read this scene as 80% risk, but the fusion gives its branch zero weight, so it did not influence this forecast. These conditions most closely resemble Mumbai 2017-07-18 (Heavy, 66 mm), at 79% similarity. Confidence is 95%, based on high agreement between the weather and satellite branches, 79% similarity to the closest past event and good input data quality."
}
```

**Feature attributions** (signed, in fused-risk probability units)

| Rank | Feature | Contribution | Share | Impact |
|---|---|---|---|---|
| 1 | Rainfall, past 24h | +0.1261 | +30.0% | Critical |
| 2 | Longitude | +0.0609 | +14.5% | Medium |
| 3 | Rainfall, past 3 days | +0.0504 | +12.0% | Medium |
| 4 | Rainfall, past 7 days | +0.0502 | +12.0% | Medium |
| 5 | Humidity | +0.0360 | +8.6% | Medium |

Baseline 0.6666 + Σ attributions = 0.9973, the explained risk — the
attribution reconstructs the prediction exactly.

**Satellite explanation** — satellite risk 0.80, *substantial* coverage
(17.1% of the scene), two regions: `cloud_cluster_area_1` in the
northern-western quadrant and `cloud_cluster_area_2` in the
northern-eastern quadrant. Overlay: `reports/gradcam/gradcam_mumbai_2017-07-17.png`.

**Confidence factors** — model agreement 0.99, historical similarity 0.79,
data quality Good (1.00 usable scene, no imputed inputs).

**Caveat carried in the payload** — location and calendar terms absorb 25%
of the attribution here. They are real inputs and appear in the table, but
are not voiced as causes.

---

## Example 2 — Mumbai, 23 June 2005

An early-monsoon case where the two branches *disagree*: the weather
branch is emphatic, the satellite scene much less so.

- **Prediction:** Heavy Rainfall, 99.7% probability, CRITICAL
- **Confidence:** High (84.2%) — lower than Example 1 precisely because
  agreement is weaker (0.71 against 0.99)
- **Top drivers:** Rainfall past 24h +0.1372 (**+33.5%, Critical**),
  Rainfall past 3 days +0.0653 (+15.9%, High), Rainfall past 7 days
  +0.0593 (+14.5%, Medium)
- **Satellite:** risk 0.57, *substantial* coverage (22.0%), regions on the
  eastern edge and northern part of the scene
- **Closest analogue:** Kerala 2019-08-07 (Heavy, 65 mm) at 65.2%

This is the case that shows confidence is doing real work: the same
headline probability as Example 1, but 11 points less confidence, traced
directly to branch disagreement and a weaker historical match.

---

## Example 3 — Mumbai, 4 August 2019

Shows the model weighing evidence in both directions.

- **Prediction:** Heavy Rainfall, 79.6% probability, CRITICAL
- **Confidence:** High (83.5%)
- **Top drivers:** Rainfall past 24h **+0.1264 (+30.2%)** raising risk,
  against Rainfall past 30 days **−0.0808 (−19.3%)** pulling it back — a
  wet day inside an otherwise dry month
- **Satellite:** risk 0.96 but only *isolated* coverage (2.9% of scene) —
  the model is confident yet its attention is concentrated in two small
  southern areas
- **Closest analogue:** Mumbai 2007-06-30 (Heavy, 153 mm) at 72.8%

The negative attribution is the useful part: the narrative reports that
the dry preceding month held risk down, rather than listing only the
evidence that supports the alarm.

---

## What these examples do *not* show

Three limits are visible in the output above and are stated in every
payload's `caveats` block:

1. **The satellite branch carried zero weight.** In all three cases the
   heatmap explains the Phase 4 model's own same-day reading, not the
   fused forecast. Note Example 3, where the satellite model read 0.96
   while the fused prediction was 0.80 — the imagery genuinely did not
   participate.
2. **Location and calendar terms absorb 22–25% of attribution.** That is a
   real property of the Phase 3 forest, surfaced rather than hidden.
3. **No "Extreme Rainfall" example exists.** One such region-day survives
   into the fusion dataset and none fall in the test period, so no
   Extreme-class explanation could be produced or validated. The
   `event_names` mapping supports it; the data does not.

## Reproducing

```bash
backend/.venv/bin/python -m explainability.reports.writer --cases 3
backend/.venv/bin/python -m explainability.reports.writer --region Mumbai --date 2017-07-17
```

Artifacts land in `reports/shap/` (bar, waterfall, summary),
`reports/gradcam/` (overlays), and `reports/xai_report_v1.txt` /
`reports/xai_explanations_v1.json`.

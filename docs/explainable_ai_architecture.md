# VARUNA AI — Explainable AI Architecture (Phase 6)

Code: `explainability/` · Reports: `reports/shap/`, `reports/gradcam/`,
`reports/xai_report_v1.txt`, `reports/xai_explanations_v1.json`

## 1. What this layer answers

Every served prediction must be able to answer *"why did the AI decide
this?"* in three registers at once:

| Register | Question | Backend |
|---|---|---|
| Numeric | Which measurements moved the score, and by how much? | SHAP |
| Spatial | Which parts of the satellite scene mattered? | Grad-CAM (Captum) |
| Narrative | What does that mean in plain language? | Explanation generator |

Plus two supporting signals: how confident the system is and *why*, and
which past event today most resembles.

## 2. The compatibility problem SHAP had to solve

The Phase 5 engine is a **pipeline, not a single estimator**, so there is
no one model to hand to SHAP:

```
risk = w · P_weather(weather features) + (1 − w) · P_satellite(scene)
```

The shipped v1 fusion is `WeightedFusion` with **w = 1.00**, and it reads
only the two upstream probability vectors. Measured directly:

```
perturbing humidity_pct    by +50%  →  Δ fused output = 0.000000
perturbing cloud_density   by +50%  →  Δ fused output = 0.000000
perturbing satellite_prob  by +0.30 →  Δ fused output = 0.000000   (w = 1)
perturbing weather_prob    by +0.30 →  Δ fused output = 0.197561
```

Running SHAP against the blender would therefore return **exactly zero
for every meteorological feature**. Any non-zero number printed beside
"Humidity" would be invented — precisely what the phase brief forbids.

**Resolution — attribute by the chain rule.** An input's effect on the
decision runs through whichever branch consumes it, so SHAP is applied to
that branch and scaled by the branch's exact fusion weight:

```
∂risk/∂humidity = w · ∂P_weather/∂humidity
```

This is exact, not an approximation, because the blend is linear in the
branch outputs and `w` is a known constant.

## 3. SHAP workflow

```
fusion bundle ──► which strategy was selected?
   │
   ├── WeightedFusion   → TreeSHAP on the Phase 3 forest, × w
   │                      + analytic satellite term, × (1 − w)
   ├── FeatureFusion    → TreeSHAP on the fusion tree ensemble (22 features)
   └── NeuralFusion     → KernelSHAP over the 22-feature vector
   │
   ▼
signed contributions in fused-risk probability units
   │  share = contribution / Σ|contributions|
   │  impact label from the measured share (Critical ≥30%, High ≥15%, Medium ≥5%)
   ▼
ranked attributions + unattributed_inputs + notes
```

Three properties are enforced and tested:

- **Additivity.** `base_value + Σ contributions == predicted_risk`. The
  explanation reconstructs the number it claims to explain.
- **Weight scaling.** Halving the weather weight halves every weather
  attribution — verified in the test suite.
- **No silent gaps.** Inputs the selected model cannot read are listed in
  `unattributed_inputs` with the reason, never scored.

### Region coordinates

The Phase 3 forest reads latitude and longitude as features, but the
fusion dataset aggregates to region-days and drops them. Filling them from
the global median put roughly 20% of the attribution mass on a location
belonging to no region, and materially changed predictions — one Mumbai
case moved from 68.9% to 99.7% once corrected. The explainer therefore
resolves each region's real centroid from the Phase 2 weather record
before scoring, and only falls back to a median when no coordinate exists.
Values that *are* median-filled are flagged `imputed: true`.

### Out-of-fold divergence

Rows in `fusion_dataset.parquet` carry **out-of-fold** upstream
probabilities (Phase 5's leakage control), while serving — and this
explainer — use the **shipped** Phase 3 artifact. The two differ on
historical rows. The payload reports both (`predicted_risk` vs
`served_risk`) and states the reason rather than letting the gap look like
an explanation error.

## 4. Grad-CAM workflow

```
satellite scene ──► Phase 4 checkpoint (custom_cnn)
                      │  target layer read from the checkpoint's
                      │  `gradcam_target_layer`, not guessed
                      ▼
              Captum LayerGradCam, target = predicted class
                      ▼
        14×14 attribution grid ──bilinear──► scene resolution
                      ▼
          normalise to [0,1] → threshold → connected components
                      ▼
   named regions (cloud_cluster_area_N) + bbox + coverage + position
```

Phase 4 recorded the target layer at training time precisely so this stage
does not have to guess it. The attribution grid is the layer's own
resolution; bilinear upsampling smooths those values for display without
implying pixel-level precision the attribution does not have.

Regions are ranked by `area × intensity`, filtered by a minimum area, and
named `cloud_cluster_area_1..N`. The overlay writes red for high influence
and blue for low, with each region boxed.

**The honest caveat this layer carries:** when the fusion gives the
satellite branch weight 0, the heatmap explains *the satellite model's own
same-day reading*, not the fused forecast. The narrative says so outright.

## 5. Explanation generation pipeline

```
   SHAP ──────────┐
   Grad-CAM ──────┤
   Historical ────┼──► ExplanationGenerator ──► UnifiedExplanation
   Confidence ────┘                              │
                                                 ├─ narrative (templates)
                                                 ├─ caveats
                                                 └─ artifacts (figures)
```

`templates.py` decides **phrasing**, never **content**. No sentence
asserts that a feature matters; a feature is named only because SHAP
measured it into the top ranks for that specific prediction.

Two classes of driver are deliberately kept out of prose while remaining
in the attribution table:

- **median-filled inputs** — describing them asserts something about the
  region that was never observed;
- **geometry and calendar terms** (latitude, longitude, season, wind
  direction) — the model genuinely uses them, but "elevated longitude" is
  not a cause an operator can act on.

When those terms absorb ≥10% of the attribution, the payload says so. On
current data they carry 22–25%, which is itself a finding about the Phase 3
model worth surfacing.

## 6. Confidence explanation

Rebuilt component by component from the Phase 5 blend:

| Factor | Weight | Source |
|---|---|---|
| Prediction strength | 0.5 | winning class probability |
| Model agreement | 0.3 | `1 − |weather_score − satellite_score|` |
| Historical similarity | 0.2 | closest analogue's similarity |

**Data quality** is measured rather than asserted: the usable fraction of
the satellite scene (swath gaps excluded) discounted by how many model
inputs had to be median-filled. Components that cannot be computed are
dropped and the remaining weights renormalised, so an absent signal never
silently counts as zero.

## 7. Historical similarity

`historical_explainer.py` wraps the Phase 5 matcher. Events are named from
what the record contains — region, date, observed category, observed
rainfall — for example *"Kerala 2019-08-07 (Heavy, 65 mm)"*. A popular
nickname the dataset cannot substantiate is never invented. Replaying a
day that is itself a reference event excludes its own entry, so it cannot
match itself at 100% and inflate confidence.

## 8. Unified output contract

```json
{
  "prediction": "Heavy Rainfall",
  "probability": 0.9973,
  "confidence": "High",
  "risk_level": "CRITICAL",
  "top_features": ["Rainfall, past 24h", "Rainfall, past 3 days"],
  "satellite_regions": ["cloud_cluster_area_1", "cloud_cluster_area_2"],
  "explanation": "Conditions indicate a critical risk of high-impact rainfall …"
}
```

The payload also carries `feature_attributions`, `confidence_explanation`,
`satellite_explanation`, `historical_explanation`, `shap_detail`,
`caveats` and `artifacts`. `caveats` is a first-class field: an
explanation that hides its own limits is worse than none, because it
invites more trust than the evidence supports.

## 9. Running it

```bash
backend/.venv/bin/python -m explainability.reports.writer
backend/.venv/bin/python -m explainability.reports.writer --region Kerala --date 2018-08-15
```

Writes the JSON payload, the text report, per-prediction SHAP bar and
waterfall figures, the Grad-CAM overlay, and the model-wide SHAP summary.

## 10. Known limits

- **The satellite branch carries zero weight in v1**, so Grad-CAM
  currently explains the satellite model rather than the fused decision.
  It becomes decision-relevant the moment a fusion with `w < 1` is
  selected — no code change needed.
- **Extreme is unexplainable because it is unpredictable here.** One such
  region-day survives into the fusion dataset and none are in test, so no
  attribution for that class can be validated.
- **KernelSHAP is sampled**, so the neural-fusion path returns
  approximations bounded by its background set; the two tree paths are
  exact.
- Attribution explains *the model*, not the atmosphere. A feature ranked
  highly is one this forest leans on — not a claim about physical
  causation.

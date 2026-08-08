# VARUNA AI — Hybrid AI Architecture (Phase 5)

Code: `ai_models/fusion_model/` · Bundle:
`ai_models/saved_models/varuna_fusion_model_v1.pkl` · Metadata:
`ai_models/saved_models/model_info.json` · Report:
`reports/hybrid_model_report_v1.txt`

## 1. What the fusion is for

Phase 3 forecasts rainfall category from weather; Phase 4 reads cloud
structure from satellite imagery. Phase 5 combines them into one risk
engine so that a warning is supported by both the atmospheric state and
what the sky actually looks like.

## 2. Compatibility analysis — the two models do not answer the same question

This was checked before any fusion code was written, and it determined
the whole design.

| | Phase 3 (weather) | Phase 4 (satellite) |
|---|---|---|
| Input | day-T weather + rainfall history | day-T MODIS scene |
| Target | rainfall category at **T+1** | rainfall category at **T** |
| Granularity | one point (lat/lon) | one region |
| Output | `weather_risk_score` = P(heavy)+P(extreme) | `satellite_risk_score` = P(heavy)+P(extreme) |

The two risk scores share a vocabulary but not a referent: one is a
next-day forecast, the other a same-day nowcast. Averaging them directly
would silently blend a forecast with a hindcast.

**Resolution.** The fusion target is the **T+1 region-day category**,
matching the early-warning mission. Both branches contribute their day-T
outputs as *evidence about tomorrow*: the weather branch natively, the
satellite branch as an indicator of current convective state. Spatially,
everything is aggregated to the region-day — weather variables averaged
across the region's points, rainfall taken as the regional maximum,
which is how Phase 4 labelled its scenes.

## 3. Data flow

```
data/processed/datasets/training_dataset.parquet   (Phase 2, point-days)
        │  region-day aggregation, past-only rain history, season encoding
        │  Phase 3 scored per point → regional worst case kept
        ▼
   weather region-days ──┐
                         ├── inner join on (region, date) ──► fusion_dataset.parquet
   satellite region-days ┘
        ▲
        │  scene statistics (Phase 4 preprocessing) + Phase 4 model risk score
data/processed/features/satellite_features.parquet
```

The join is inner: a region-day survives only if both modalities
genuinely observed it. Satellite coverage is the binding constraint.

## 4. Leakage control — why the weather score is regenerated

The shipped Phase 3 artifact was fitted on most of the fusion period. Its
score on that period is memorised, not predicted:

| split | AUC of `weather_risk_score` vs the T+1 target |
|---|---|
| fusion train | 1.000 |
| fusion validation | 1.000 |
| fusion test | 0.875 |

Trained on that feature, all three fusion approaches reached a validation
macro-F1 of **1.000** — they had simply learned to threshold a memorised
value — and then collapsed on test. The feature is therefore regenerated
**out-of-fold by walk-forward retraining**: the record is split into
folds, and each fold is scored by a fresh clone of the Phase 3 estimator
fitted only on strictly earlier days. Region-days inside the initial
warm-up block get no score and leave the dataset.

After the correction the weather score reads AUC 0.849 / 0.860 / 0.822
across train / validation / test — informative, no longer memorised.

The satellite score was checked the same way and left in-sample: its AUC
is already stable across splits (0.842 / 0.792 / 0.799), showing no
comparable memorisation. Walk-forward retraining of the CNN was not run —
it would mean retraining the vision model once per fold. Both figures are
recomputed on every run and printed in the report's CAVEATS block, so they
cannot drift away from the data.

At serving time no such correction is needed: predicting a genuinely
future day is out-of-sample by construction, so `predict.py` uses the
full Phase 3 and Phase 4 artifacts.

## 5. Fusion strategies compared

All three implement `fit(...)` / `predict_proba(...)`, so training
compares them uniformly and serving does not care which one won.

**Approach 1 — weighted late fusion** (`WeightedFusion`). A convex blend
of the two upstream probability vectors,
`w · weather + (1 − w) · satellite`, with `w` swept on validation. Nothing
is learned beyond one scalar, which makes it the honest floor the trained
approaches must clear. A sweep that lands on `w = 0` or `w = 1` means the
blend collapsed to a single modality — the report says so explicitly.

**Approach 2 — feature-level fusion** (`FeatureFusion`). Weather features,
satellite scene statistics and both upstream risk scores concatenated into
one vector, then a Random Forest or XGBoost classifier, with
inverse-frequency sample weights.

**Approach 3 — neural fusion network** (`NeuralFusion`). Separate MLP
encoders per modality feeding a shared fusion layer, so the network can
weigh the modalities per sample rather than applying one global weight.
On a dataset of ~10³ rows it is the most likely to overfit; it is included
because the phase brief asks for the comparison, and its result is
reported as measured.

Selection is on **validation macro-F1**, with a single held-out test
evaluation afterwards — the same discipline as Phases 3 and 4. Two
single-modality baselines (`weather_only`, `satellite_only`) are scored on
the identical test period so the report can state what fusion actually
bought.

## 6. Model communication

The fusion never imports model internals. It consumes two things from
each upstream branch — a class-probability vector and a scalar risk score —
plus the interpretable scene statistics. That contract is what lets Phase
4 swap architectures, or Phase 3 be retrained, without touching fusion
code.

`HybridRainfallModel` in `predict.py` adapts the bundle to the
project-wide `RainfallModel` interface for the Phase 7 backend.

## 7. Confidence

Confidence blends three signals, each in [0, 1], with weights in
`FusionConfig`:

- **prediction probability** (0.5) — the winning class probability
- **model agreement** (0.3) — `1 − |weather_score − satellite_score|`;
  a warning both branches support is better founded
- **historical similarity** (0.2) — similarity to the closest past
  high-impact event (see `historical_match.py`)

Components that cannot be computed are dropped and the remaining weights
renormalised, so an absent signal never counts as zero.

## 8. Historical pattern matching

`historical_match.py` compares the current region-day against past
high-impact events in standardized feature space and reports the closest
analogues with a similarity percentage
(`similarity = exp(-distance / scale)`, scale = median pairwise distance).
Events are named from what the record contains — region, date, observed
category, observed rainfall — never from a popular nickname the dataset
cannot substantiate. When a historical day is replayed, its own entry is
excluded so it cannot match itself at 100%.

## 9. Environment note — OpenMP

torch, xgboost and scikit-learn each vendor a copy of `libomp`. This
pipeline drives torch (Phase 4 scene scoring) and xgboost (fusion
candidate) in one process, and on macOS the two runtimes collide: the
process segfaults inside `DMatrix` construction or deadlocks, whichever
initialises second, in *both* import orders. `KMP_DUPLICATE_LIB_OK` does
not help. `ai_models/fusion_model/__init__.py` therefore pins
`OMP_NUM_THREADS=1` before any numeric library loads, which costs nothing
on a dataset this size.

## 10. Known limits

Stated in full in the report's CAVEATS block, regenerated from measured
values on every run. The load-bearing ones:

- **Extreme is not measurable.** One such region-day survives into the
  fusion dataset and none land in the test period. Needs IMD gridded /
  GPM IMERG data — the same gap carried from Phases 3 and 4.
- **The comparison is underpowered.** Selection and verdict each rest on
  ~130 region-days holding ~15 high-impact days; a few macro-F1 points
  are noise.
- **Fusion has not yet beaten the weather branch alone** on held-out
  data. See the report for the current measured outcome.

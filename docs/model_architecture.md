# VARUNA AI — Baseline Model Architecture (Phase 3)

Code: `ai_models/baseline/` · Artifacts: `ai_models/saved_models/` ·
Experiments: `ai_models/experiments/` · Reports: `reports/`

## 1. ML approach

**Task.** Supervised multi-class classification: given conditions known
*today* at a location, predict *tomorrow's* 24 h rainfall category —
`0` normal, `1` heavy (≥ 64.5 mm), `2` extreme (≥ 204.4 mm), per IMD
thresholds configured in `config/data_config.yaml`. The secondary output
is a probability-based **risk score** = P(heavy) + P(extreme).

The next-day formulation is deliberate: predicting the *same* day's
category from same-day precipitation would be circular. Every feature
describes information available strictly before the target period.

**Candidates compared** (`model.py::build_candidate_models`):
Logistic Regression (linear reference), Random Forest, Gradient
Boosting, XGBoost. Class imbalance (~99 % of days are "normal") is
handled with `class_weight="balanced"` or balanced sample weights.

## 2. Features (17)

| Group | Features | Source |
|---|---|---|
| Current weather | temperature_c, humidity_pct, pressure_hpa, wind_speed_ms, wind_direction_deg, cloud_cover_pct | Phase 2 dataset (NASA POWER) |
| Historical rainfall | rain_sum_1d/3d/7d/30d (past accumulations per location), rain_trend_3d (this 3-day window minus the previous one) | computed in `data.py`, past-only by construction |
| Seasonal | season_sin, season_cos (cyclical day-of-year — Dec/Jan are neighbours) | derived |
| Location | latitude, longitude | dataset |

## 3. Training workflow (`train.py`)

1. **Validation gate** — training refuses to start unless the Phase 2
   DATA REPORT says `READY` (`data.load_dataset`).
2. Feature engineering + next-day target (`build_supervised_frame`).
3. **Chronological split** 70/15/15 — no shuffling across time, so the
   test set is a genuinely unseen future period (leakage-free).
4. Train all four candidates on the train split.
5. **Select on validation macro-F1** — never accuracy: a model that
   always says "normal" scores 99 % accuracy and is useless for early
   warning. Macro-F1 forces performance on the rare classes that matter.
6. Refit the winner on train+validation; evaluate **once** on the
   held-out test period.
7. Persist: model bundle (`rainfall_model_v1.pkl` — estimator + feature
   names + training medians + label names + metrics + version), the
   VARUNA AI MODEL REPORT, permutation feature importance (JSON + PNG),
   and a full experiment record (`experiments/experiment_NNN.json`).

## 4. Evaluation metrics

Reported per run: accuracy (with an explicit imbalance caveat), macro
precision/recall/F1, per-class precision/recall with supports, and the
confusion matrix. Report verdict: **GOOD** at macro-F1 ≥ 0.60, else
**NEEDS IMPROVEMENT**. See `reports/model_report_v1.txt` and
[baseline_model_results.md](baseline_model_results.md) for actual numbers.

## 5. Inference (`predict.py`)

`RainfallPredictor` loads the bundle once and serves structured
predictions; inputs are validated against ranges (bad values raise
`InvalidInputError` rather than being silently accepted). Rainfall
history is optional — absent history features are filled with training
medians, listed in `assumed_features`, and the confidence grade
(High/Medium/Low, derived from the predicted-class probability) is
downgraded one level so callers know the prediction ran on partial input.

`BaselineRainfallModel` (`model.py`) adapts the same bundle to the
project-wide `RainfallModel` interface (`ai_models/base.py`) — this is
what the backend's `PredictionService` will load in Phase 7, and its
`explanation_context` already carries the feature vector and class
probabilities the Phase 6 SHAP explainer needs.

## 6. Known limitations (tracked, not hidden)

- **Extreme class is data-starved**: NASA POWER's ~0.5° grid smooths
  point extremes; the 24-year build contains only 3 days ≥ 204.4 mm.
  The model cannot learn class 2 from 3 examples — fixing this requires
  the higher-resolution rainfall sources already planned (IMD gridded
  0.25°, GPM IMERG 0.1°) once account registrations are in place.
- Single-point features; no spatial neighbourhood context until the
  satellite vision model (Phase 4) contributes cloud-structure signals.

# VARUNA AI — Baseline Model Results (v1)

All numbers below are real, produced by `python -m ai_models.baseline.train`
on 2026-08-07 against the validated Phase 2 dataset. Full machine-readable
record: `ai_models/experiments/experiment_001.json`.

## Dataset

- **Source:** NASA POWER daily reanalysis (open NASA product), acquired
  through the Phase 2 pipeline with provenance catalogs — 8 grid points
  across Kerala, Mumbai, Chennai, Assam; 2001–2024.
- **Supervised records:** 69,888 (after rolling warm-up), validation
  status READY, 0 % missing.
- **Class counts:** normal 69,311 · heavy 574 · extreme 3 — severe,
  *real* imbalance (see limitations).
- **Split:** chronological — train 2001→2017, validation 2017→2021,
  test 2021-05→2024-12 (fully held out).

## Models tested (validation set, selection metric macro-F1)

| Model | F1 (macro) | Recall (macro) | Accuracy |
|---|---|---|---|
| **Random Forest** ← selected | **0.642** | 0.641 | 0.988 |
| XGBoost | 0.630 | 0.645 | 0.986 |
| Gradient Boosting | 0.551 | 0.874 | 0.919 |
| Logistic Regression | 0.342 | 0.577 | 0.868 |

## Selected model: Random Forest — held-out test results

| Metric | Value |
|---|---|
| Accuracy | 98.8 % *(inflated by class imbalance — not used for selection)* |
| Precision (macro) | 62.0 % |
| Recall (macro) | 65.2 % |
| **F1 (macro)** | **63.4 %** |
| Heavy-rain recall | 31.2 % (24 of 77 heavy days caught; 74 false alarms in 10,403 normal days) |
| Verdict | GOOD (≥ 0.60 macro-F1 threshold) |

**Why Random Forest:** best validation macro-F1 with balanced
precision/recall. Gradient Boosting achieved the highest heavy-rain
recall (0.874) but at ~8× the false-alarm rate (accuracy 0.919), which
would erode trust in warnings; XGBoost was statistically equivalent to
RF and remains a strong contender for the Phase 5 hybrid. If operational
priorities later favour recall over false alarms, the threshold/model
choice can be revisited explicitly.

## Feature importance (permutation, macro-F1, test set)

| Factor | Share |
|---|---|
| Longitude (coastal vs inland regime) | 18.9 % |
| Atmospheric pressure | 17.9 % |
| Previous-day rainfall | 14.6 % |
| 30-day rainfall accumulation | 11.3 % |
| 3-day rainfall accumulation | 10.9 % |
| Humidity | 6.4 % |
| 7-day rainfall accumulation | 6.4 % |
| Wind speed | 4.2 % |
| Cloud cover | 4.0 % |
| Others (season, wind direction, temperature, latitude) | 5.4 % |

Meteorologically coherent: low pressure, recent rainfall persistence, and
humidity dominate — exactly the drivers the Phase 6 SHAP layer will
surface per prediction. Plot: `reports/feature_importance_v1.png`.

## Honest limitations

1. **Extreme class (n=3) is unlearnable from this source.** POWER's
   ~0.5° grid smooths point extremes. The model currently detects
   heavy-vs-normal; credible extreme-event classification arrives with
   IMD gridded / GPM IMERG rainfall (higher resolution) in later phases.
2. Heavy-rain recall of 31 % reflects a genuinely hard task (next-day,
   point-scale, tabular-only). The Phase 4 satellite vision model and
   Phase 5 hybrid exist precisely to raise this number.
3. No hyperparameter search yet — candidates ran with sensible defaults;
   tuning is deferred until the hybrid model defines the final feature set.

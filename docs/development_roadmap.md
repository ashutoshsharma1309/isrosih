# VARUNA AI — Development Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Architecture & foundation | ✅ Done |
| 2 | Data acquisition & preprocessing pipeline | ✅ Done |
| 3 | Baseline tabular rainfall model | ✅ Done |
| 4 | Satellite image intelligence model | ✅ Done |
| 5 | Hybrid prediction system | ✅ Done |
| 6 | Explainable AI layer | ✅ Done |
| 7 | Backend API integration | Next |
| 8 | Frontend dashboard build-out | Pending |
| 9 | Integration & testing | Pending |
| 10 | Deployment & SIH presentation | Pending |

## Phase 1 — Foundation (complete)

Project structure, FastAPI backend skeleton with stable API contracts,
React + Tailwind + Leaflet frontend shell, `RainfallModel`/`Explainer`
interfaces, PostGIS schema, Docker orchestration, documentation.

## Phase 2 — Data pipeline (complete)

Implemented in `backend/data_pipeline/` (see docs/preprocessing_pipeline.md):
collectors for MOSDAC/NASA/IMD/IMERG/ERA5 with provenance catalogs;
satellite/weather/rainfall preprocessors; IMD-threshold label generation
(0/1/2); dataset merger producing `training_dataset.parquet`; validation
system emitting the VARUNA AI DATA REPORT (READY / NOT READY);
`config/data_config.yaml`; 22 pipeline tests.

Operational follow-ups (need account registrations, run alongside Phase 3):
downloading 2–3 monsoon seasons of real INSAT/IMD/ERA5 data for the pilot
regions, plus EDA notebooks on the acquired data.

## Phase 3 — Baseline tabular model (complete)

Implemented in `ai_models/baseline/` (see docs/model_architecture.md and
docs/baseline_model_results.md): real training data acquired via NASA
POWER through the Phase 2 pipeline (70k records, 2001–2024, 4 regions);
four candidates compared (LogReg, RF, GB, XGBoost) with chronological
splits and imbalance handling; Random Forest selected (test macro-F1
0.634); artifact `ai_models/saved_models/rainfall_model_v1.pkl`
implements the `RainfallModel` interface; permutation feature importance,
experiment tracking, structured prediction CLI, 16 tests.

Known gap carried forward: extreme-class (≥204.4 mm) training data —
needs IMD gridded / GPM IMERG resolution.

## Phase 4 — Vision model (complete)

Implemented in `ai_models/satellite_model/` (see
docs/satellite_model_architecture.md, docs/satellite_data_processing.md):
1,144 real MODIS scenes (NASA GIBS True Color + Band31 IR, 2001–2024)
labeled with observed region-day rainfall outcomes; chronological splits,
class-weighted loss and augmentation; Grad-CAM target layer recorded in
the checkpoint for Phase 6; interpretable scene features (cloud density,
cold-top fraction, growth rate, dispersion) exported for the Phase 5
fusion. INSAT 30-min sequences remain the documented MOSDAC upgrade path.

Trained artifacts: `ai_models/saved_models/satellite_model_v1.pt`
(custom_cnn, test macro-F1 0.860), report, curves, confusion matrix,
sample-prediction grid, and `satellite_features.parquet`.

Two gaps recorded rather than papered over:

- **The three-way architecture comparison did not run.** ResNet-18 and
  ViT-B/16 are implemented but were never evaluated — first because
  `torchvision` could not fetch ImageNet weights (SSL CA bundle), then
  because the 8 GB development machine could not train them. `custom_cnn`
  was selected as the only candidate that ran. See
  docs/satellite_model_architecture.md for the one-command reproduction.
- **Extreme is untrainable here.** Only 3 Extreme scenes exist and the
  chronological split puts all 3 in train, so validation and test contain
  none — the class that matters most for an extreme-rainfall warning
  system cannot be scored. Same data gap as Phase 3.

## Phase 5 — Hybrid system (complete)

Implemented in `ai_models/fusion_model/` (see docs/hybrid_ai_architecture.md
and docs/prediction_pipeline.md). Unified region-day dataset joining Phase 2
weather with Phase 4 scene features; three fusion approaches compared
(weighted late fusion, feature-level RF/XGBoost, two-encoder neural fusion)
against both single-modality baselines on the same held-out period;
confidence from prediction strength + branch agreement + historical
similarity; `historical_match.py` for nearest past analogues; bundle,
`model_info.json`, VARUNA AI HYBRID MODEL REPORT and 35 tests.

Two findings changed the design and are worth carrying forward:

- **The two models answer different questions.** Phase 3 forecasts T+1 per
  point; Phase 4 classifies T per region. The fusion target is the T+1
  region-day, so the satellite branch contributes current convective state,
  not a same-day answer.
- **The Phase 3 score leaked.** Scored with the shipped artifact, it had
  AUC 1.000 in-sample versus 0.875 out-of-sample, and every fusion approach
  reached a meaningless validation macro-F1 of 1.000. It is now generated
  out-of-fold by walk-forward retraining. The satellite score was checked
  the same way and shows no such memorisation.

Honest outcome: **fusion has not yet beaten the weather branch alone** on
held-out data — the validation weight sweep chose w=1.00, discarding the
satellite branch. See `reports/hybrid_model_report_v1.txt`. The blocking
constraints are the class imbalance (one Extreme region-day survives into
the dataset, none in test) and a test period holding ~15 high-impact days,
which is too small to separate the approaches.

Carried forward to Phase 6+: probability calibration; a recall- or
cost-weighted selection metric (the current macro-F1 winner trades away
event recall, the wrong trade for early warning); `PredictionService`
wiring so the 501s disappear (Phase 7).

## Phase 6 — Explainability (complete)

Implemented in `explainability/` (see docs/explainable_ai_architecture.md
and docs/ai_explanation_examples.md): SHAP attribution over the fusion
pipeline, Captum Grad-CAM over the Phase 4 CNN, natural-language narrative
generation, confidence decomposition, historical analogues, and 31 tests.

The design decision that shaped the phase: **the Phase 5 engine is a
pipeline, not a single estimator.** The shipped `WeightedFusion` (w=1.00)
reads only the two upstream probability vectors — perturbing humidity or
cloud density changes its output by exactly 0.000000. Running SHAP against
it would return zero for every meteorological feature, so attribution
follows the chain rule instead: SHAP on the branch that reads each input,
scaled by that branch's exact fusion weight. Attributions are verified to
reconstruct the prediction (`base + Σ contributions == predicted_risk`).

Two corrections came out of building it:

- **Region coordinates were being median-filled.** The Phase 3 forest
  reads latitude/longitude, which the region-day aggregation drops. Filling
  from the global median put ~20% of attribution on a location belonging to
  no region and shifted one case from 68.9% to 99.7%. The explainer now
  resolves each region's real centroid.
- **Location and calendar terms absorb 22–25% of attribution.** Real, and
  now disclosed — they stay in the attribution table but are not voiced as
  causes, because "elevated longitude" is not something an operator can act
  on.

Carried forward: the satellite branch has zero fusion weight in v1, so
Grad-CAM currently explains the Phase 4 model's own reading rather than the
fused decision. It becomes decision-relevant with no code change the moment
a fusion with w < 1 is selected.

## Phase 7 — Backend integration

- Alembic migrations, real DB reads/writes, alert threshold engine,
  scheduled batch predictions, prediction persistence.

## Phase 8 — Dashboard

- Risk-zone choropleth from live predictions, satellite + Grad-CAM overlay
  viewer, SHAP explanation panel, historical analysis, alert feed.

## Phase 9 — Integration & testing

- End-to-end tests on a replayed historical event (e.g. Kerala 2018);
  load testing; failure-mode review (missing data, stale imagery).

## Phase 10 — Deployment & presentation

- Cloud deployment; demo script replaying a documented extreme event with
  live explanations; SIH pitch materials.

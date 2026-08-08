# VARUNA AI — Development Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Architecture & foundation | ✅ Done |
| 2 | Data acquisition & preprocessing pipeline | ✅ Done |
| 3 | Baseline tabular rainfall model | ✅ Done |
| 4 | Satellite image intelligence model | ✅ Done |
| 5 | Hybrid prediction system | Next |
| 6 | Explainable AI layer | Pending |
| 7 | Backend API integration | Pending |
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
labeled with observed region-day rainfall outcomes; custom CNN vs
ResNet-18 transfer vs ViT-B/16 linear probe compared with chronological
splits, class-weighted loss, and augmentation; Grad-CAM target layer
recorded in the checkpoint for Phase 6; interpretable scene features
(cloud density, cold-top fraction, growth rate, dispersion) exported for
the Phase 5 fusion. INSAT 30-min sequences remain the documented MOSDAC
upgrade path.

## Phase 5 — Hybrid system

- Fusion of tabular + vision outputs; probability calibration;
  `PredictionService` loads the artifact and the 501s disappear.

## Phase 6 — Explainability

- SHAP explainer for tabular features; Grad-CAM renderer for imagery;
  narrative generator; payloads conform to the `Explanation` schema.

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

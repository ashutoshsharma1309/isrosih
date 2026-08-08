# VARUNA AI — Project Requirements

**Problem statement:** SIH260006 (ISRO) — *"Explainable AI model to predict
high-impact rain events using satellite data."*

## 1. Goal

A software-only, government-grade disaster intelligence platform that
predicts high-impact rainfall events from satellite and meteorological
data, and — critically — explains every prediction in terms a disaster
management official can act on.

## 2. Functional requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Ingest satellite imagery (ISRO MOSDAC INSAT products; NASA Earth observation datasets) | Must |
| FR2 | Ingest meteorological time series: rainfall, temperature, humidity, pressure, wind speed, cloud parameters | Must |
| FR3 | Predict probability of a high-impact rain event for a location and lead time (1–72 h) | Must |
| FR4 | Classify risk as low / moderate / heavy / extreme with a confidence score | Must |
| FR5 | Explain tabular predictions with SHAP feature attributions | Must |
| FR6 | Explain image-based predictions with Grad-CAM heatmaps over satellite imagery | Must |
| FR7 | Generate a plain-language narrative for every prediction | Must |
| FR8 | Interactive dashboard: India map with risk zones, satellite visualisation, prediction detail, explanation panels, historical analysis | Must |
| FR9 | Generate and display human-readable early-warning alerts when risk thresholds are crossed | Must |
| FR10 | Persist all predictions with their explanations for auditability | Should |
| FR11 | Historical rainfall analysis view | Should |

## 3. Non-functional requirements

- **No fabricated output** — the system serves real model output or clearly
  reports that no model/data is available (HTTP 501, empty lists). This is
  a hard rule at every phase.
- **Explainability first** — a prediction without an explanation payload is
  considered incomplete.
- **Separation of concerns** — AI code (`ai_models/`, `explainability/`,
  `satellite_processing/`) never imports application code, and the backend
  consumes models only through the `RainfallModel` interface.
- **Reproducibility** — raw data is immutable; every processed dataset and
  model artifact is versioned and traceable to its inputs.
- **Deployability** — the full stack runs with one `docker compose up`.
- **Scalability** — stateless API; models loaded once per process; spatial
  queries indexed via PostGIS.

## 4. Out of scope

- Hardware/sensor deployment (software-only solution).
- SMS/phone alert delivery infrastructure (alerts are generated and
  displayed; delivery integration is a future extension).
- Global coverage — the system targets India and its meteorological
  subdivisions.

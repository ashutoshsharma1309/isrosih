# VARUNA AI

**An Explainable AI-Based Extreme Rainfall Intelligence & Early Warning
System Using Satellite Data**

Built for Smart India Hackathon — problem statement **SIH260006 (ISRO)**:
*"Explainable AI model to predict high-impact rain events using satellite data."*

---

## Problem

High-impact rainfall events (cloudbursts, extreme monsoon spells) cause
floods, landslides, and loss of life across India. Existing forecasts are
either too coarse or act as black boxes — disaster management authorities
receive a number without the reasoning behind it, which undermines trust
and slows response.

## Solution

VARUNA AI is a disaster intelligence platform that:

1. **Ingests** ISRO INSAT satellite imagery (MOSDAC), NASA Earth
   observation data, and meteorological time series.
2. **Predicts** the probability of a high-impact rain event per region and
   lead time (1–72 h), classified low → extreme with a confidence score.
3. **Explains every prediction** — SHAP feature attributions for weather
   drivers, Grad-CAM heatmaps showing *which cloud structures* in the
   satellite image drove the prediction, and a plain-language narrative.
4. **Warns** — generates human-readable early-warning alerts when risk
   thresholds are crossed, displayed on an interactive India risk map.

Explainability is the core differentiator: the platform answers not just
*"will it rain heavily?"* but *"why does the AI believe that?"*.

## Features

- AI rainfall prediction engine (probability + risk class + confidence)
- Satellite image intelligence (cloud density, formation, movement)
- Explainable AI layer (SHAP + Grad-CAM + narrative)
- Interactive disaster monitoring dashboard (India map, risk zones,
  satellite visualisation, explanation panels, historical analysis)
- Early-warning alert generation with audit trail

## Technology stack

| Layer | Technology |
|---|---|
| Backend API | Python, FastAPI, Pydantic |
| ML | PyTorch (vision), scikit-learn (tabular), pandas, NumPy, OpenCV |
| Explainability | SHAP, Grad-CAM |
| Database | PostgreSQL + PostGIS |
| Frontend | React, Tailwind CSS, Leaflet |
| Deployment | Docker, docker-compose |

## Architecture

```
data sources → satellite_processing → data/ + PostGIS
                                        ↓
                   ai_models (tabular · vision · hybrid)
                                        ↓
                   explainability (SHAP · Grad-CAM · narrative)
                                        ↓
                   backend (FastAPI REST API)
                                        ↓
                   frontend (React dashboard)
```

Strict boundaries: AI packages never import application code; the backend
consumes models only through the `RainfallModel` interface
([ai_models/base.py](ai_models/base.py)). Full details in
[docs/system_architecture.md](docs/system_architecture.md).

## Repository layout

```
VARUNA-AI/
├── backend/               FastAPI application (api / core / schemas / services / db)
├── frontend/              React + Tailwind + Leaflet dashboard
├── ai_models/             ML models: tabular, vision, hybrid (+ base interfaces)
├── explainability/        SHAP, Grad-CAM, narrative generation
├── satellite_processing/  Data acquisition (MOSDAC, NASA) & preprocessing
├── data/                  raw / processed / external datasets (git-ignored)
├── database/              PostGIS schema + migrations
├── notebooks/             EDA and model experiments
├── docs/                  Requirements, architecture, data spec, roadmap
└── deployment/            docker-compose orchestration
```

## Getting started

**Full stack (Docker):**

```bash
cp .env.example .env
docker compose -f deployment/docker-compose.yml up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:3000
```

**Backend only (local dev):**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload   # http://localhost:8000/docs
pytest                          # run the test suite
```

**Frontend only (local dev):**

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

> **Honest-by-design:** prediction endpoints return HTTP 501 until trained
> models are deployed (Phases 3–6). The platform never fabricates
> predictions or alerts — the dashboard shows real system status.

## Project phases

Phases 1–6 are complete: architecture, data pipeline, baseline tabular
model, satellite vision model, the hybrid fusion engine, and the
explainability layer. Phase 7 (backend API integration) is next. The full
ten-phase plan is in
[docs/development_roadmap.md](docs/development_roadmap.md).

| Phase | Deliverable | Trained artifact |
|---|---|---|
| 3 | Tabular model — next-day category from weather | `rainfall_model_v1.pkl` (test macro-F1 0.634) |
| 4 | Satellite CNN — same-day category from a MODIS scene | `satellite_model_v1.pt` (test macro-F1 0.860) |
| 5 | Hybrid fusion — next-day risk from both branches | `varuna_fusion_model_v1.pkl` (see report) |
| 6 | Explainability — SHAP, Grad-CAM, narrative | no artifact; explains 3–5 on demand |

Model artifacts are git-ignored; regenerate them with
`python -m ai_models.<package>.train`, or distribute via releases/DVC.
Every run writes a report to [reports/](reports/) whose CAVEATS block
names what the numbers do not show — currently that Extreme rainfall is
unmeasurable on this data, and that fusion has not yet beaten the weather
branch alone on held-out days.

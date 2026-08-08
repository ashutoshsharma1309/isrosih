# VARUNA AI — System Architecture

## 1. High-level view

```
             ┌────────────────────────────────────────────────────────┐
             │                    DATA SOURCES                        │
             │  MOSDAC (INSAT-3D/3DR) · NASA Earthdata (IMERG,        │
             │  MERRA-2) · IMD historical rainfall · boundaries       │
             └───────────────┬────────────────────────────────────────┘
                             │  scheduled acquisition (Phase 2)
                             ▼
┌───────────────────────────────────────────┐
│        satellite_processing/              │
│  source clients → georeference → crop →   │
│  normalise → temporal stacks → tensors    │
└───────┬───────────────────────────┬───────┘
        │ data/raw, data/processed  │ metadata
        ▼                           ▼
┌───────────────┐        ┌─────────────────────┐
│   data/       │        │ PostgreSQL+PostGIS  │
│  (files)      │        │ regions · obs ·     │
└───────┬───────┘        │ images · preds ·    │
        │                │ alerts              │
        ▼                └─────────┬───────────┘
┌───────────────────────────────┐  │
│          ai_models/           │  │
│  tabular (GBM)  vision (CNN)  │  │
│        └── hybrid fusion ──┐  │  │
└────────────┬───────────────┼──┘  │
             │ ModelOutput   │     │
             ▼               │     │
┌───────────────────────────────┐  │
│        explainability/        │  │
│  SHAP · Grad-CAM · narrative  │  │
└────────────┬──────────────────┘  │
             │ Explanation payload │
             ▼                     ▼
┌────────────────────────────────────────────┐
│           backend/ (FastAPI)               │
│  /predictions  /alerts  /health  /model-info│
└────────────────────┬───────────────────────┘
                     │ REST (JSON)
                     ▼
┌────────────────────────────────────────────┐
│         frontend/ (React + Leaflet)        │
│  risk map · satellite view · explanation   │
│  panels · alerts · historical analysis     │
└────────────────────────────────────────────┘
```

## 2. Layer responsibilities and boundaries

| Layer | Owns | May depend on |
|---|---|---|
| `satellite_processing/` | Data acquisition + preprocessing | nothing internal |
| `ai_models/` | Training + inference; exposes `RainfallModel` interface | processed data |
| `explainability/` | SHAP, Grad-CAM, narratives; consumes `ModelOutput.explanation_context` | `ai_models.base` |
| `backend/` | API contract, persistence, alert thresholds, orchestration | `ai_models.base`, `explainability.base` |
| `frontend/` | Visualisation only; talks exclusively to the REST API | backend API |

**Dependency rule:** arrows point inward only. The backend depends on AI
*interfaces*, never on training code; AI packages never import the backend.
This is what lets model teams and app teams work in parallel between phases.

## 3. Backend internal structure (clean architecture)

```
backend/app/
├── main.py                 # app factory, middleware, lifespan
├── core/                   # config (env-driven), logging
├── api/v1/                 # HTTP layer: routers + endpoints only
├── schemas/                # Pydantic contracts (single source of truth)
├── services/               # business logic; seam to ai_models/
└── db/                     # SQLAlchemy session management
```

Endpoints never contain business logic; services never contain HTTP
concerns. `PredictionService` is the only place that will touch model
artifacts.

## 4. Prediction request flow (target state, Phase 7)

1. `POST /api/v1/predictions` with location + horizon.
2. `PredictionService` gathers latest weather features (DB) and the most
   recent preprocessed satellite tensor for the region.
3. Hybrid model returns `ModelOutput` (probability, risk level, confidence,
   explanation context).
4. Explainability layer computes SHAP attributions + Grad-CAM heatmap and
   composes the narrative.
5. Prediction + explanation persisted to `predictions` (JSONB payload).
6. If probability crosses the configured threshold, an alert row is created.
7. Response returned to the dashboard; the map, explanation panel, and
   alert feed update.

## 5. Key design decisions

- **Interface-first AI integration** — API contract and `RainfallModel`
  interface fixed in Phase 1 so Phases 3–8 can proceed in parallel.
- **PostGIS** — spatial matching of points to regions, footprint queries
  for satellite scenes, and GeoJSON generation for the map come from the
  database rather than app code.
- **Explanations stored, not recomputed** — the JSONB payload preserves
  exactly what the user was shown, which matters for auditing warnings.
- **501 until real** — unserved capabilities fail loudly instead of
  returning demo data; the dashboard renders true system status.

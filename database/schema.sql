-- =====================================================================
-- VARUNA AI — PostgreSQL + PostGIS schema (v1)
-- Applied automatically by the postgis container on first start
-- (mounted into /docker-entrypoint-initdb.d by docker-compose).
-- Later migrations are managed with Alembic in database/migrations/.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- Monitored regions (states, districts, coastal zones). The geometry
-- lets the dashboard draw risk zones and lets queries match points to
-- regions spatially.
CREATE TABLE IF NOT EXISTS regions (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,          -- e.g. 'Kerala'
    admin_level TEXT NOT NULL DEFAULT 'state', -- state | district | zone
    boundary    GEOMETRY(MULTIPOLYGON, 4326),
    centroid    GEOMETRY(POINT, 4326)
);
CREATE INDEX IF NOT EXISTS idx_regions_boundary ON regions USING GIST (boundary);

-- Time-series meteorological observations feeding the tabular model.
CREATE TABLE IF NOT EXISTS weather_observations (
    id              BIGSERIAL PRIMARY KEY,
    region_id       INTEGER REFERENCES regions(id),
    location        GEOMETRY(POINT, 4326) NOT NULL,
    observed_at     TIMESTAMPTZ NOT NULL,
    temperature_c   REAL,
    humidity_pct    REAL CHECK (humidity_pct BETWEEN 0 AND 100),
    pressure_hpa    REAL,
    wind_speed_ms   REAL CHECK (wind_speed_ms >= 0),
    cloud_cover_pct REAL CHECK (cloud_cover_pct BETWEEN 0 AND 100),
    rainfall_mm     REAL CHECK (rainfall_mm >= 0),
    source          TEXT NOT NULL              -- e.g. 'IMD', 'MERRA-2'
);
CREATE INDEX IF NOT EXISTS idx_obs_region_time ON weather_observations (region_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_obs_location ON weather_observations USING GIST (location);

-- Catalog of downloaded satellite imagery (files themselves live in data/,
-- object storage in production; the DB stores metadata + footprint).
CREATE TABLE IF NOT EXISTS satellite_images (
    id           BIGSERIAL PRIMARY KEY,
    product      TEXT NOT NULL,                -- e.g. 'INSAT-3D IMG_TIR1'
    source       TEXT NOT NULL,                -- 'MOSDAC' | 'NASA'
    captured_at  TIMESTAMPTZ NOT NULL,
    footprint    GEOMETRY(POLYGON, 4326),
    storage_path TEXT NOT NULL,                -- relative path under data/
    processed    BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (product, captured_at, storage_path)
);
CREATE INDEX IF NOT EXISTS idx_sat_captured ON satellite_images (captured_at DESC);

-- Every model prediction is persisted for auditability and for the
-- dashboard's historical analysis view.
CREATE TABLE IF NOT EXISTS predictions (
    id            BIGSERIAL PRIMARY KEY,
    region_id     INTEGER REFERENCES regions(id),
    location      GEOMETRY(POINT, 4326) NOT NULL,
    generated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    horizon_hours INTEGER NOT NULL,
    probability   REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    risk_level    TEXT NOT NULL CHECK (risk_level IN ('low','moderate','heavy','extreme')),
    confidence    REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    model_version TEXT NOT NULL,
    -- Full explanation payload (SHAP attributions, Grad-CAM references,
    -- narrative) as produced by the explainability layer.
    explanation   JSONB
);
CREATE INDEX IF NOT EXISTS idx_pred_region_time ON predictions (region_id, generated_at DESC);

-- Early-warning alerts generated when predictions cross risk thresholds.
CREATE TABLE IF NOT EXISTS alerts (
    id            BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES predictions(id),
    region_id     INTEGER REFERENCES regions(id),
    risk_level    TEXT NOT NULL CHECK (risk_level IN ('low','moderate','heavy','extreme')),
    probability   REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    message       TEXT NOT NULL,               -- human-readable warning incl. AI reasoning
    valid_from    TIMESTAMPTZ NOT NULL,
    valid_until   TIMESTAMPTZ NOT NULL,
    issued_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active     BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts (is_active, issued_at DESC);

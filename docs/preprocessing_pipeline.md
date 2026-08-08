# VARUNA AI — Preprocessing Pipeline

Code: `backend/data_pipeline/` · Config: `config/data_config.yaml` ·
Orchestrator: `python -m data_pipeline.run` (from `backend/`)

```
            COLLECT                    PREPROCESS                 MERGE / VALIDATE
┌─────────────────────────┐  ┌───────────────────────────┐  ┌──────────────────────────┐
│ SatelliteCollector      │  │ SatelliteProcessor        │  │ DatasetMerger            │
│  MOSDAC orders / URLs   │─▶│  decode → corruption check│─▶│  weather ⋈ rainfall      │
│                         │  │  → resize 224² → normalize│  │  (hour + rounded coords) │
│ WeatherCollector        │  │  → float32 .npy + index   │  │  + nearest satellite     │
│  ERA5 (cdsapi / files)  │─▶│ WeatherProcessor          │  │    image within window   │
│                         │  │  UTC ts → dedup → bounded │  │            │             │
│ RainfallCollector       │  │  interpolation → outlier  │  │            ▼             │
│  IMERG / IMD files      │─▶│  flags → z-score (stats   │  │ DataValidator            │
│                         │  │  persisted)               │  │  8 checks → DATA REPORT  │
│  → data/raw/* +         │  │ RainfallProcessor         │  │  → READY / NOT READY     │
│    provenance catalogs  │  │  clean → region map →     │  │                          │
└─────────────────────────┘  │  IMD category labels 0/1/2│  └──────────────────────────┘
                             └───────────────────────────┘
```

## Transformations by stage

**Satellite** (`preprocessors/satellite_processor.py`)
- Corruption detection (undecodable/empty files recorded and skipped)
- Grayscale/BGR → explicit channel dim, RGB ordering
- Resize to `satellite.image_size`² (INTER_AREA), min-max normalize to [0, 1]
- Output: float32 `(H, W, C)` `.npy` (CNN/ViT-ready; `to_tensor_layout` → CHW)
  plus a JSONL index with shape/dtype/normalization per image

**Weather** (`preprocessors/weather_processor.py`)
- Timestamps parsed to timezone-aware UTC; unparseable records dropped
- Duplicates removed on (timestamp, lat, lon)
- Missing values: linear time interpolation bounded by
  `weather.max_interpolation_gap` — long gaps are *left missing*, never invented
- Outliers: |z| > `weather.outlier_zscore` → flagged (`is_outlier`), not deleted
  (extreme weather records are the signal, not noise)
- Z-score normalization into `*_norm` columns; fitted stats persisted to
  `data/metadata/weather_normalization_stats.json` for identical
  inference-time transforms

**Rainfall** (`preprocessors/rainfall_processor.py`)
- Cleaning: invalid timestamps/coordinates, negative rainfall, duplicates
- Region mapping against configured bounding boxes
- Label generation (IMD 24 h categories, thresholds from config):
  `0` normal < 64.5 mm · `1` heavy ≥ 64.5 mm · `2` extreme ≥ 204.4 mm
- Labels written to `data/labels/rainfall_labels.parquet`

**Merge** (`merger.py`)
- Inner join only — a record exists in the training set only when weather
  AND rainfall genuinely coincide; labels are never imputed
- Satellite image linked per record by nearest capture time within
  `merger.max_image_time_gap_hours`; otherwise `satellite_image_path = None`
- Output: `data/processed/datasets/training_dataset.parquet`

**Validate** (`validators/data_validator.py`)
- Required columns, minimum records, missing-value ceiling, duplicates,
  coordinate/timestamp validity, label integrity, class presence,
  satellite reference loadability (sampled)
- Report: `data/metadata/validation_report.txt` — dataset is **READY**
  only when every blocking check passes; Phase 3 training must refuse
  NOT READY datasets

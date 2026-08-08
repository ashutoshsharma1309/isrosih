# VARUNA AI — Training Dataset Schema

Canonical file: `data/processed/datasets/training_dataset.parquet`
(produced by `DatasetMerger`; human preview in `training_dataset_preview.csv`).

One row = one (hour, grid point) where weather features and a rainfall
label both truly exist.

## Columns

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime (UTC) | Observation hour (floored) |
| `latitude` | float | WGS84, rounded to `merger.coordinate_precision` |
| `longitude` | float | WGS84, rounded to `merger.coordinate_precision` |
| `region` | string \| null | Configured region containing the point (e.g. `Kerala`) |
| `temperature_c` | float | Surface air temperature, °C |
| `humidity_pct` | float | Relative humidity, % (0–100) |
| `pressure_hpa` | float | Mean sea-level pressure, hPa |
| `wind_speed_ms` | float | 10 m wind speed, m/s |
| `wind_direction_deg` | float | Wind direction, degrees (0–360) |
| `cloud_cover_pct` | float | Total cloud cover, % (0–100) |
| `precipitation_mm` | float | Hourly precipitation, mm |
| `<feature>_norm` | float | Z-scored version of each feature above (stats in `data/metadata/weather_normalization_stats.json`) |
| `is_outlier` | bool | Any feature exceeded the z-score threshold (flag, not exclusion) |
| `satellite_image_path` | string \| null | Repo-relative path to the model-ready `.npy` image nearest in time (≤ `merger.max_image_time_gap_hours`); null when no image qualifies |
| `rainfall_mm` | float | 24 h rainfall accumulation — regression target |
| `rainfall_category` | int | Classification label: `0` normal · `1` heavy (≥ 64.5 mm) · `2` extreme (≥ 204.4 mm) |

## Supporting artifacts

| File | Purpose |
|---|---|
| `data/labels/rainfall_labels.parquet` | Standalone label table (timestamp, location, region, mm, category) |
| `data/processed/images/<product>/*.npy` | Normalized float32 `(H, W, C)` satellite arrays |
| `data/processed/images/<product>_index.jsonl` | Per-image provenance: source path, shape, dtype, normalization |
| `data/metadata/<domain>_catalog.jsonl` | Raw acquisition catalogs (SHA-256, source, product, timestamp) |
| `data/metadata/weather_normalization_stats.json` | Fitted normalization parameters — reuse at inference time |
| `data/metadata/validation_report.txt` | Latest VARUNA AI DATA REPORT (READY / NOT READY) |

## Contract with Phase 3 (model training)

- Train/validation splits must be **time-based** (no shuffling across time)
  to avoid leakage.
- `rainfall_category` is the classification target; `rainfall_mm` supports
  regression or ordinal formulations.
- Rows with `is_outlier = True` are included by default — training decides
  their treatment explicitly.
- Training must abort if the latest validation report is NOT READY.

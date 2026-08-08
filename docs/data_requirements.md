# VARUNA AI — Data Requirements (Phase 2 input)

## 1. Satellite imagery

| Source | Product | Use | Access |
|---|---|---|---|
| ISRO MOSDAC | INSAT-3D/3DR Imager — TIR1/TIR2 brightness temperature | Cloud-top temperature → convective intensity | Registered account (free); order/API via mosdac.gov.in |
| ISRO MOSDAC | INSAT-3D Cloud Top Pressure/Temperature, Cloud Mask | Cloud density & structure features | Same |
| ISRO MOSDAC | Hydro-Estimator / IMSRA rainfall | Satellite-derived rainfall labels & validation | Same |
| NASA Earthdata | GPM IMERG (half-hourly, 0.1°) | High-quality precipitation ground truth | Earthdata token |
| NASA Earthdata | MERRA-2 reanalysis | Gap-filling meteorological fields | Earthdata token |

Notes:
- INSAT imager cadence is ~30 min — sequences of consecutive frames give the
  cloud-movement signal for the vision model (temporal stacking).
- Target region: Indian mainland + coastal waters; store scene footprints in
  PostGIS for spatial queries.

## 2. Meteorological time series

Required per location/region, at least hourly where available:
temperature, relative humidity, mean sea-level pressure, wind speed,
cloud cover, and observed rainfall (the label source).

Candidate sources: IMD gridded rainfall (0.25°, daily; long history for
training labels), IMD AWS/ARG station feeds, MERRA-2/ERA5 reanalysis for
historical feature reconstruction.

## 3. Reference data (`data/external/`)

- India state/district boundaries (Survey of India / GADM) → `regions` table.
- Historical extreme-event catalog (e.g. Kerala 2018 floods, Chennai 2015)
  for case-study evaluation — the demo should replay a real documented event.

## 4. Label definition (critical modelling decision)

"High-impact rain event" must be defined quantitatively before training.
Working definition to validate in Phase 2 against IMD categories:

- **Heavy**: 64.5–115.5 mm / 24 h
- **Very heavy**: 115.6–204.4 mm / 24 h
- **Extremely heavy**: > 204.4 mm / 24 h (primary positive class)

Class imbalance will be severe (extreme events are rare) — evaluation must
use precision/recall, PR-AUC, and event-based hit/miss rates, never accuracy.

## 5. Storage & governance

- `data/raw/` — immutable, exactly as downloaded, named
  `<source>/<product>/<YYYY-MM-DD>/...`.
- `data/processed/` — model-ready arrays + Parquet feature tables, each with
  a provenance sidecar (source files, processing params, code version).
- All datasets git-ignored; catalog lives in PostgreSQL
  (`satellite_images`, `weather_observations`).
- Credentials only via environment variables (`.env`, never committed).

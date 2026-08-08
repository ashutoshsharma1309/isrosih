# VARUNA AI — Data Sources

Every source below is real and officially distributed. The pipeline never
generates or simulates data; collectors ingest actual deliveries or fail
with clear errors.

## 1. Satellite data

| Source | Products | Data type | Purpose | Access |
|---|---|---|---|---|
| **ISRO MOSDAC** (primary) — mosdac.gov.in | INSAT-3D/3DR Imager: TIR1 (10.8 µm), TIR2 (12.0 µm), VIS (0.65 µm), MIR (3.9 µm); derived Cloud Top Pressure/Temp; Hydro-Estimator rainfall | HDF5/GeoTIFF rasters, ~30 min cadence | Cloud density, cloud-top temperature (convective intensity), formation and movement patterns — inputs to the vision model | Free registered account; order/download workflow → `SatelliteCollector.collect(order_dir=...)` |
| **NASA Earthdata** (secondary) | Worldview/GIBS imagery, MODIS/VIIRS cloud products | Rasters | Supplementary coverage and validation | Earthdata token (`NASA_EARTHDATA_TOKEN`) → `collect(urls=[...])` |

## 2. Rainfall data (label source)

| Source | Product | Data type | Purpose | Access |
|---|---|---|---|---|
| **NASA GPM IMERG** | Half-hourly / daily gridded precipitation, 0.1° | NetCDF/HDF5 | High-quality precipitation ground truth; label generation | Earthdata token, GES DISC URLs → `RainfallCollector.collect(urls=[...])` |
| **IMD** | Gridded daily rainfall (0.25°), station records | GRD/CSV | Long historical record over India; extreme-event labels; validation against official categories | Download from imdpune.gov.in → `collect(local_dir=...)` |

## 3. Weather / atmospheric data

| Source | Product | Data type | Purpose | Access |
|---|---|---|---|---|
| **ERA5** (Copernicus CDS) | Single-level reanalysis: 2 m temperature, 2 m dewpoint (→ humidity), MSL pressure, 10 m u/v wind (→ speed + direction), total cloud cover, total precipitation | NetCDF, hourly, 0.25° | The tabular feature set for the baseline model | Free CDS account; `WeatherCollector.request_era5(...)` (cdsapi) or web download → `collect(local_dir=...)` |

## Provenance

Every acquired file is checksummed (SHA-256) and recorded in
`data/metadata/<domain>_catalog.jsonl` with source, product, size, and
acquisition timestamp. Raw files are immutable once ingested.

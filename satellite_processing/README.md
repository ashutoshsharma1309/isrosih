# satellite_processing/

**Phase 2 update:** data acquisition and preprocessing are implemented in
[`backend/data_pipeline/`](../backend/data_pipeline/) (collectors,
preprocessors, validators) — that package is the canonical pipeline. See
[docs/preprocessing_pipeline.md](../docs/preprocessing_pipeline.md).

This package is reserved for the advanced geospatial work arriving with
the vision model (Phases 4–5), which goes beyond generic preprocessing:

- `sources/` — product-specific readers for INSAT HDF5/NetCDF containers
  (channel extraction, calibration to brightness temperature)
- `preprocessing/` — georeferencing, region-of-interest cropping from
  scene footprints, and temporal frame stacking for cloud-motion features

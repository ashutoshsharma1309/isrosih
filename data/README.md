# data/

Datasets are git-ignored (only this README and `.gitkeep` markers are
tracked). Populated by the Phase 2 acquisition pipeline in
`satellite_processing/`.

- `raw/` — immutable downloads exactly as received from MOSDAC, NASA
  Earthdata, IMD, etc. Never edited in place.
- `processed/` — cleaned, aligned, model-ready datasets derived from
  `raw/` by reproducible pipeline code (no manual edits).
- `external/` — third-party reference data (district boundaries,
  historical flood records, station metadata).

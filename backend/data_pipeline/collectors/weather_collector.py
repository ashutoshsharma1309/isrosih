"""
Weather data collector — ERA5 climate reanalysis (Copernicus CDS).

Responsibilities:
- Acquire atmospheric variables (temperature, humidity, pressure, wind,
  cloud cover, precipitation) as NetCDF files
- Store raw files unchanged under data/raw/weather/<product>/
- Maintain a provenance catalog (data/metadata/weather_catalog.jsonl)

ERA5 is served by the Copernicus Climate Data Store, which uses the
`cdsapi` client with a personal key (~/.cdsapirc). `request_era5()` builds
and submits a real CDS request when the client is configured, and fails
with setup instructions when it is not. `collect(local_dir=...)` ingests
NetCDF files downloaded through the CDS web interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from data_pipeline.collectors.base_collector import BaseCollector, CollectionError, RawAsset
from data_pipeline.config import DataConfig, Region

logger = logging.getLogger(__name__)

#: ERA5 single-level variable names for the features in data_config.yaml.
ERA5_VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",  # → relative humidity (derived in preprocessing)
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",  # u,v → speed + direction (derived)
    "total_cloud_cover",
    "total_precipitation",
)


class WeatherCollector(BaseCollector):
    domain = "weather"

    def __init__(self, config: DataConfig) -> None:
        super().__init__(config, config.paths.raw_weather)

    def collect(
        self,
        *,
        local_dir: str | Path | None = None,
        product: str = "ERA5_SINGLE_LEVELS",
        source: str = "ERA5",
        **_: Any,
    ) -> list[RawAsset]:
        """Ingest ERA5 NetCDF files downloaded via the CDS web interface."""
        if local_dir is None:
            raise CollectionError(
                "Nothing to collect: pass local_dir=<downloaded ERA5 NetCDF files>, "
                "or use request_era5() for the CDS API workflow."
            )
        local_path = Path(local_dir)
        if not local_path.is_dir():
            raise CollectionError(f"Directory does not exist: {local_path}")

        assets = [
            self.ingest_local_file(path, source=source, product=product, subdir=product)
            for path in sorted(local_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".nc", ".nc4", ".grib"}
        ]
        if not assets:
            raise CollectionError(f"No NetCDF/GRIB files found in {local_path}")
        logger.info("[weather] collected %d asset(s) for %s", len(assets), product)
        return assets

    def request_era5(self, region: Region, year: int, months: list[int]) -> RawAsset:
        """Submit a real ERA5 retrieval via the CDS API and catalog the result.

        Requires the `cdsapi` package and a configured ~/.cdsapirc key
        (https://cds.climate.copernicus.eu/how-to-api).
        """
        try:
            import cdsapi  # optional dependency; only needed for API retrieval
        except ImportError as exc:
            raise CollectionError(
                "cdsapi is not installed. Run `pip install cdsapi` and configure "
                "~/.cdsapirc, or download ERA5 NetCDF manually and use collect(local_dir=...)."
            ) from exc

        lon_min, lat_min, lon_max, lat_max = region.bbox
        dest_dir = self.raw_dir / "ERA5_SINGLE_LEVELS"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"era5_{region.name.lower()}_{year}_{min(months):02d}-{max(months):02d}.nc"

        request = {
            "product_type": "reanalysis",
            "variable": list(ERA5_VARIABLES),
            "year": str(year),
            "month": [f"{m:02d}" for m in months],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            # CDS expects area as [north, west, south, east]
            "area": [lat_max, lon_min, lat_min, lon_max],
            "format": "netcdf",
        }
        logger.info("[weather] submitting ERA5 request for %s %d/%s", region.name, year, months)
        try:
            client = cdsapi.Client()
            client.retrieve("reanalysis-era5-single-levels", request, str(dest))
        except Exception as exc:  # cdsapi raises plain Exceptions
            dest.unlink(missing_ok=True)
            raise CollectionError(f"ERA5 retrieval failed: {exc}") from exc

        return self._register(
            dest,
            source="ERA5",
            product="ERA5_SINGLE_LEVELS",
            extra={"region": region.name, "year": year, "months": months},
        )

"""
NASA GIBS/Worldview satellite imagery collector.

GIBS (Global Imagery Browse Services) is NASA's open, no-authentication
imagery service. VARUNA AI uses it as the first end-to-end-acquirable
satellite source while MOSDAC INSAT registration is pending:

- MODIS Terra Corrected Reflectance (True Color) — visible cloud
  structure, daily since 2000, ~10:30 local overpass.
- MODIS Terra Brightness Temperature Band 31 (~11 µm thermal IR) —
  cloud-top temperature proxy (cold, high convective tops appear dark).

Every downloaded scene is stored unchanged under data/raw/satellite/
and cataloged with provenance like all other sources.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

from data_pipeline.collectors.base_collector import BaseCollector, CollectionError, RawAsset
from data_pipeline.config import DataConfig, Region

logger = logging.getLogger(__name__)

_SNAPSHOT_URL = "https://wvs.earthdata.nasa.gov/api/v1/snapshot"

LAYER_TRUE_COLOR = "MODIS_Terra_CorrectedReflectance_TrueColor"
LAYER_IR_BT = "MODIS_Terra_Brightness_Temp_Band31_Day"

#: Minimum bbox span (degrees) — cloud systems are synoptic-scale, so
#: small city bboxes are padded out to give the model spatial context.
_MIN_SPAN_DEG = 4.0


def padded_bbox(region: Region) -> tuple[float, float, float, float]:
    """Region bbox expanded to at least _MIN_SPAN_DEG on each axis."""
    lon_min, lat_min, lon_max, lat_max = region.bbox
    lon_pad = max(_MIN_SPAN_DEG - (lon_max - lon_min), 0) / 2
    lat_pad = max(_MIN_SPAN_DEG - (lat_max - lat_min), 0) / 2
    return (lon_min - lon_pad, lat_min - lat_pad, lon_max + lon_pad, lat_max + lat_pad)


class GibsCollector(BaseCollector):
    domain = "satellite"

    def __init__(self, config: DataConfig, image_size: int = 512) -> None:
        super().__init__(config, config.paths.raw_satellite)
        self.image_size = image_size

    def collect(
        self,
        *,
        region: Region,
        date: str,
        layer: str = LAYER_TRUE_COLOR,
        **_: Any,
    ) -> list[RawAsset]:
        """Download one snapshot (region, date, layer) → data/raw/satellite/."""
        return [self._fetch(region, date, layer)]

    def collect_many(
        self,
        requests: list[tuple[Region, str, str]],
        *,
        max_workers: int = 8,
    ) -> tuple[list[RawAsset], list[tuple[str, str, str]]]:
        """Concurrent bulk download.

        Returns (successful_assets, failures[(region, date, layer)]).
        Failures are collected, not raised — a bulk historical pull should
        survive individual missing scenes, and the caller decides whether
        the yield is sufficient.
        """
        assets: list[RawAsset] = []
        failures: list[tuple[str, str, str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._fetch, region, date, layer): (region.name, date, layer)
                for region, date, layer in requests
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    assets.append(future.result())
                except CollectionError as exc:
                    logger.warning("Snapshot failed %s: %s", key, exc)
                    failures.append(key)
        logger.info("[satellite] GIBS bulk: %d ok, %d failed", len(assets), len(failures))
        return assets, failures

    # ------------------------------------------------------------------
    def _fetch(self, region: Region, date: str, layer: str) -> RawAsset:
        lon_min, lat_min, lon_max, lat_max = padded_bbox(region)
        params = {
            "REQUEST": "GetSnapshot",
            "TIME": date,
            "BBOX": f"{lat_min},{lon_min},{lat_max},{lon_max}",
            "CRS": "EPSG:4326",
            "LAYERS": layer,
            "WRAP": "day",
            "FORMAT": "image/jpeg",
            "WIDTH": self.image_size,
            "HEIGHT": self.image_size,
        }
        subdir = self.raw_dir / "NASA_GIBS" / layer
        subdir.mkdir(parents=True, exist_ok=True)
        dest = subdir / f"{region.name.lower()}_{date}.jpg"

        if dest.exists() and dest.stat().st_size > 0:
            # Already downloaded in a previous run — idempotent re-runs.
            return self._register(dest, source="NASA_GIBS", product=layer,
                                  extra={"region": region.name, "date": date, "cached": True})

        transport = httpx.HTTPTransport(retries=3)
        try:
            with httpx.Client(transport=transport, timeout=90.0) as client:
                response = client.get(_SNAPSHOT_URL, params=params)
                response.raise_for_status()
                if not response.headers.get("content-type", "").startswith("image/"):
                    raise CollectionError(
                        f"GIBS returned non-image content for {region.name} {date} {layer}"
                    )
                dest.write_bytes(response.content)
        except httpx.HTTPError as exc:
            dest.unlink(missing_ok=True)
            raise CollectionError(f"GIBS snapshot failed ({region.name}, {date}): {exc}") from exc

        return self._register(
            dest,
            source="NASA_GIBS",
            product=layer,
            extra={"region": region.name, "date": date,
                   "bbox": [lon_min, lat_min, lon_max, lat_max]},
        )

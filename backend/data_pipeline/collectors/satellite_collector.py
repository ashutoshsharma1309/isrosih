"""
Satellite data collector — ISRO MOSDAC (primary), NASA Earthdata (secondary).

Responsibilities:
- Acquire INSAT-3D/3DR imagery products (IR, visible, cloud products)
- Store raw files unchanged under data/raw/satellite/<product>/
- Maintain a provenance catalog (data/metadata/satellite_catalog.jsonl)

Acquisition modes (both bring in REAL data; nothing is simulated):

1. `collect(order_dir=...)` — MOSDAC distributes most INSAT products via
   an order/download workflow rather than an open REST API. Operators
   place an order on mosdac.gov.in, download the delivery, and point this
   collector at the directory; every file is checksummed and cataloged.

2. `collect(urls=[...])` — direct HTTP download for sources that expose
   stable URLs (e.g. NASA Earthdata, which authenticates with a bearer
   token from the NASA_EARTHDATA_TOKEN environment variable).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from data_pipeline.collectors.base_collector import BaseCollector, CollectionError, RawAsset
from data_pipeline.config import DataConfig

logger = logging.getLogger(__name__)

#: INSAT imager channels/products we target (see docs/data_sources.md).
KNOWN_PRODUCTS = (
    "INSAT-3D_IMG_TIR1",  # thermal IR 10.8 µm — cloud-top temperature
    "INSAT-3D_IMG_TIR2",  # thermal IR 12.0 µm
    "INSAT-3D_IMG_VIS",   # visible 0.65 µm — daytime cloud structure
    "INSAT-3D_IMG_MIR",   # mid-IR 3.9 µm
    "INSAT-3D_CTP",       # derived cloud-top pressure/temperature
    "INSAT-3D_HEM",       # hydro-estimator rainfall
)


class SatelliteCollector(BaseCollector):
    domain = "satellite"

    def __init__(self, config: DataConfig) -> None:
        super().__init__(config, config.paths.raw_satellite)

    def collect(
        self,
        *,
        order_dir: str | Path | None = None,
        urls: list[str] | None = None,
        product: str = "INSAT-3D_IMG_TIR1",
        source: str = "MOSDAC",
        **_: Any,
    ) -> list[RawAsset]:
        """Acquire satellite files from a MOSDAC order directory and/or URLs."""
        if order_dir is None and not urls:
            raise CollectionError(
                "Nothing to collect: pass order_dir=<downloaded MOSDAC order> "
                "and/or urls=[...]. This collector does not generate data."
            )
        if product not in KNOWN_PRODUCTS:
            logger.warning("Product '%s' is not in the known INSAT product list", product)

        assets: list[RawAsset] = []
        if order_dir is not None:
            assets.extend(self._ingest_order_directory(Path(order_dir), product, source))
        for url in urls or []:
            assets.append(
                self.download(
                    url,
                    source=source,
                    product=product,
                    subdir=product,
                    headers=self._auth_headers(source),
                )
            )
        logger.info("[satellite] collected %d asset(s) for %s", len(assets), product)
        return assets

    def _ingest_order_directory(self, order_dir: Path, product: str, source: str) -> list[RawAsset]:
        if not order_dir.is_dir():
            raise CollectionError(f"Order directory does not exist: {order_dir}")
        allowed = set(self.config.satellite.allowed_extensions)
        assets = []
        for path in sorted(order_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed:
                assets.append(
                    self.ingest_local_file(path, source=source, product=product, subdir=product)
                )
        if not assets:
            raise CollectionError(
                f"No files with extensions {sorted(allowed)} found in {order_dir}"
            )
        return assets

    @staticmethod
    def _auth_headers(source: str) -> dict[str, str]:
        """Bearer-token auth for NASA Earthdata URLs; MOSDAC orders are
        downloaded interactively, so no header is needed for local ingest."""
        if source.upper().startswith("NASA"):
            token = os.environ.get("NASA_EARTHDATA_TOKEN")
            if not token:
                raise CollectionError(
                    "NASA_EARTHDATA_TOKEN is not set — create a token at "
                    "https://urs.earthdata.nasa.gov and add it to .env"
                )
            return {"Authorization": f"Bearer {token}"}
        return {}

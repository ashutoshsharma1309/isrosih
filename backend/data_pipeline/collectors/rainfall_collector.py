"""
Rainfall data collector — NASA GPM IMERG and IMD rainfall datasets.

Responsibilities:
- Acquire historical rainfall records (the training label source)
- Store raw files unchanged under data/raw/rainfall/<product>/
- Maintain a provenance catalog (data/metadata/rainfall_catalog.jsonl)

Sources:
- GPM IMERG (half-hourly/daily gridded precipitation, NetCDF/HDF5) via
  NASA Earthdata authenticated URLs (GES DISC).
- IMD gridded daily rainfall (0.25°) and station records, distributed as
  downloadable files from imdpune.gov.in — ingested from a local directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from data_pipeline.collectors.base_collector import BaseCollector, CollectionError, RawAsset
from data_pipeline.config import DataConfig

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".nc", ".nc4", ".hdf5", ".h5", ".csv", ".txt", ".grd"}


class RainfallCollector(BaseCollector):
    domain = "rainfall"

    def __init__(self, config: DataConfig) -> None:
        super().__init__(config, config.paths.raw_rainfall)

    def collect(
        self,
        *,
        local_dir: str | Path | None = None,
        urls: list[str] | None = None,
        product: str = "IMD_GRIDDED_DAILY",
        source: str = "IMD",
        **_: Any,
    ) -> list[RawAsset]:
        """Acquire rainfall files from a local download directory and/or URLs."""
        if local_dir is None and not urls:
            raise CollectionError(
                "Nothing to collect: pass local_dir=<downloaded IMD/IMERG files> "
                "and/or urls=[...]. This collector does not generate data."
            )

        assets: list[RawAsset] = []
        if local_dir is not None:
            local_path = Path(local_dir)
            if not local_path.is_dir():
                raise CollectionError(f"Directory does not exist: {local_path}")
            for path in sorted(local_path.rglob("*")):
                if path.is_file() and path.suffix.lower() in _ALLOWED_EXTENSIONS:
                    assets.append(
                        self.ingest_local_file(path, source=source, product=product, subdir=product)
                    )
            if not assets:
                raise CollectionError(f"No rainfall data files found in {local_path}")

        for url in urls or []:
            assets.append(
                self.download(
                    url,
                    source=source,
                    product=product,
                    subdir=product,
                    headers=self._earthdata_headers() if source.upper().startswith("NASA") else {},
                )
            )
        logger.info("[rainfall] collected %d asset(s) for %s", len(assets), product)
        return assets

    @staticmethod
    def _earthdata_headers() -> dict[str, str]:
        token = os.environ.get("NASA_EARTHDATA_TOKEN")
        if not token:
            raise CollectionError(
                "NASA_EARTHDATA_TOKEN is not set — required for GPM IMERG downloads"
            )
        return {"Authorization": f"Bearer {token}"}

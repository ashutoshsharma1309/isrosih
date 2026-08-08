"""
Base class shared by all data collectors.

A collector's job is narrow and honest: bring REAL source files into
data/raw/<domain>/ and record provenance metadata for each one. It never
generates, simulates, or fills in data — if a source is unreachable or
credentials are missing, it raises a clear error instead.

Provenance is appended to a JSON-lines catalog in data/metadata/
(one catalog per collector), giving every raw file a checksum, source,
and acquisition timestamp — the basis for reproducible datasets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from data_pipeline.config import REPO_ROOT, DataConfig

logger = logging.getLogger(__name__)


class CollectionError(RuntimeError):
    """Raised when a source file cannot be acquired or registered."""


@dataclass(frozen=True)
class RawAsset:
    """Catalog entry describing one acquired raw file."""

    source: str  # e.g. 'MOSDAC', 'NASA_IMERG', 'ERA5'
    product: str  # e.g. 'INSAT-3D_IMG_TIR1'
    stored_path: str  # path relative to the repository root
    sha256: str
    size_bytes: int
    acquired_at: str  # ISO-8601 UTC
    extra: dict[str, Any]


class BaseCollector(ABC):
    """Common acquisition machinery: download, copy-in, checksum, catalog."""

    #: Short identifier used in logs and catalog file names, e.g. 'satellite'.
    domain: str = "base"

    def __init__(self, config: DataConfig, raw_dir: Path) -> None:
        self.config = config
        self.raw_dir = raw_dir
        self.catalog_path = config.paths.metadata / f"{self.domain}_catalog.jsonl"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Acquisition primitives
    # ------------------------------------------------------------------
    def ingest_local_file(
        self,
        source_path: str | Path,
        *,
        source: str,
        product: str,
        subdir: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RawAsset:
        """Register a file that was downloaded manually (e.g. a MOSDAC order).

        Copies it unchanged into the raw directory and catalogs it.
        """
        src = Path(source_path)
        if not src.is_file():
            raise CollectionError(f"Source file does not exist: {src}")

        dest_dir = self.raw_dir / subdir if subdir else self.raw_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
        logger.info("[%s] ingested %s -> %s", self.domain, src.name, dest_dir)
        return self._register(dest, source=source, product=product, extra=extra or {})

    def download(
        self,
        url: str,
        *,
        source: str,
        product: str,
        subdir: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 120.0,
        extra: dict[str, Any] | None = None,
    ) -> RawAsset:
        """Download a file over HTTP(S) with retries and catalog it."""
        dest_dir = self.raw_dir / subdir if subdir else self.raw_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / url.rstrip("/").split("/")[-1]

        transport = httpx.HTTPTransport(retries=3)
        try:
            with httpx.Client(
                transport=transport, timeout=timeout, follow_redirects=True
            ) as client:
                with client.stream("GET", url, headers=headers or {}) as response:
                    response.raise_for_status()
                    with dest.open("wb") as fh:
                        for chunk in response.iter_bytes(chunk_size=1 << 20):
                            fh.write(chunk)
        except httpx.HTTPError as exc:
            dest.unlink(missing_ok=True)  # never leave partial files in raw/
            raise CollectionError(f"Download failed for {url}: {exc}") from exc

        logger.info("[%s] downloaded %s (%d bytes)", self.domain, dest.name, dest.stat().st_size)
        return self._register(dest, source=source, product=product, extra={"url": url, **(extra or {})})

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------
    def _register(
        self, stored: Path, *, source: str, product: str, extra: dict[str, Any]
    ) -> RawAsset:
        asset = RawAsset(
            source=source,
            product=product,
            stored_path=str(
                stored.relative_to(REPO_ROOT) if stored.is_relative_to(REPO_ROOT) else stored
            ),
            sha256=_sha256(stored),
            size_bytes=stored.stat().st_size,
            acquired_at=datetime.now(timezone.utc).isoformat(),
            extra=extra,
        )
        with self.catalog_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(asset)) + "\n")
        return asset

    def catalog(self) -> list[RawAsset]:
        """All cataloged raw assets for this domain."""
        if not self.catalog_path.exists():
            return []
        assets = []
        with self.catalog_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    assets.append(RawAsset(**json.loads(line)))
        return assets

    # ------------------------------------------------------------------
    @abstractmethod
    def collect(self, **kwargs: Any) -> list[RawAsset]:
        """Acquire data from the collector's source(s). Implementations
        must document exactly which real source they pull from."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""
Typed access to config/data_config.yaml.

The YAML file is the single source of truth for regions, paths, and
processing parameters; this module parses it into frozen dataclasses so
the rest of the pipeline gets IDE support, validation at load time, and
no stringly-typed dict access.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# backend/data_pipeline/config.py -> backend -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "data_config.yaml"


class ConfigurationError(RuntimeError):
    """Raised when data_config.yaml is missing or structurally invalid."""


@dataclass(frozen=True)
class Region:
    name: str
    bbox: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max

    def contains(self, latitude: float, longitude: float) -> bool:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max


@dataclass(frozen=True)
class DataPaths:
    raw_satellite: Path
    raw_rainfall: Path
    raw_weather: Path
    processed_images: Path
    processed_features: Path
    processed_datasets: Path
    labels: Path
    metadata: Path

    def ensure_exist(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SatelliteConfig:
    image_size: int
    normalization: str
    allowed_extensions: tuple[str, ...]


@dataclass(frozen=True)
class WeatherConfig:
    features: tuple[str, ...]
    max_interpolation_gap: int
    outlier_zscore: float


@dataclass(frozen=True)
class RainfallConfig:
    heavy_mm: float
    extreme_mm: float
    accumulation_hours: int


@dataclass(frozen=True)
class MergerConfig:
    max_image_time_gap_hours: float
    coordinate_precision: int


@dataclass(frozen=True)
class ValidationConfig:
    max_missing_pct: float
    min_records: int


@dataclass(frozen=True)
class DataConfig:
    regions: tuple[Region, ...]
    paths: DataPaths
    satellite: SatelliteConfig
    weather: WeatherConfig
    rainfall: RainfallConfig
    merger: MergerConfig
    validation: ValidationConfig
    source_file: Path = field(default=DEFAULT_CONFIG_PATH)

    def region(self, name: str) -> Region:
        for region in self.regions:
            if region.name.lower() == name.lower():
                return region
        raise ConfigurationError(f"Region '{name}' is not defined in {self.source_file}")


def load_config(path: str | Path | None = None) -> DataConfig:
    """Load and validate the pipeline configuration.

    Resolution order: explicit argument > VARUNA_DATA_CONFIG env var >
    default config/data_config.yaml.
    """
    config_path = Path(path or os.environ.get("VARUNA_DATA_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise ConfigurationError(f"Data config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    try:
        regions = tuple(
            Region(name=r["name"], bbox=tuple(float(v) for v in r["bbox"]))
            for r in raw["regions"]
        )
        paths = DataPaths(**{k: REPO_ROOT / v for k, v in raw["paths"].items()})
        satellite = SatelliteConfig(
            image_size=int(raw["satellite"]["image_size"]),
            normalization=str(raw["satellite"]["normalization"]),
            allowed_extensions=tuple(raw["satellite"]["allowed_extensions"]),
        )
        weather = WeatherConfig(
            features=tuple(raw["weather"]["features"]),
            max_interpolation_gap=int(raw["weather"]["max_interpolation_gap"]),
            outlier_zscore=float(raw["weather"]["outlier_zscore"]),
        )
        rainfall = RainfallConfig(
            heavy_mm=float(raw["rainfall"]["heavy_mm"]),
            extreme_mm=float(raw["rainfall"]["extreme_mm"]),
            accumulation_hours=int(raw["rainfall"]["accumulation_hours"]),
        )
        merger = MergerConfig(
            max_image_time_gap_hours=float(raw["merger"]["max_image_time_gap_hours"]),
            coordinate_precision=int(raw["merger"]["coordinate_precision"]),
        )
        validation = ValidationConfig(
            max_missing_pct=float(raw["validation"]["max_missing_pct"]),
            min_records=int(raw["validation"]["min_records"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid data config {config_path}: {exc}") from exc

    if rainfall.heavy_mm >= rainfall.extreme_mm:
        raise ConfigurationError("rainfall.heavy_mm must be below rainfall.extreme_mm")

    logger.debug("Loaded data config from %s (%d regions)", config_path, len(regions))
    return DataConfig(
        regions=regions,
        paths=paths,
        satellite=satellite,
        weather=weather,
        rainfall=rainfall,
        merger=merger,
        validation=validation,
        source_file=config_path,
    )

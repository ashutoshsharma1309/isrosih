"""
NASA POWER collector — real, openly licensed meteorological data.

POWER (Prediction Of Worldwide Energy Resources, power.larc.nasa.gov)
serves NASA MERRA-2/GEOS-derived daily point data with no authentication,
which makes it the first source VARUNA AI can acquire end-to-end today
(MOSDAC and ERA5 require account registrations — see docs/data_sources.md).

Provides every configured weather feature plus daily precipitation
(PRECTOTCORR, mm/day — a 24 h accumulation, matching the IMD label
thresholds in config). Raw API responses are stored unchanged under
data/raw/weather/NASA_POWER/ and cataloged like every other source.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import numpy as np
import pandas as pd

from data_pipeline.collectors.base_collector import BaseCollector, CollectionError, RawAsset
from data_pipeline.config import DataConfig

logger = logging.getLogger(__name__)

_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

#: POWER parameter → canonical VARUNA column (with unit conversion notes).
_PARAMETERS = {
    "T2M": "temperature_c",         # °C
    "RH2M": "humidity_pct",         # %
    "PS": "pressure_hpa",           # kPa → hPa (×10)
    "WS10M": "wind_speed_ms",       # m/s
    "WD10M": "wind_direction_deg",  # degrees
    "CLOUD_AMT": "cloud_cover_pct", # %
    "PRECTOTCORR": "precipitation_mm",  # mm/day (24 h accumulation)
}
_FILL_VALUE = -999.0


class NasaPowerCollector(BaseCollector):
    domain = "weather"

    def __init__(self, config: DataConfig) -> None:
        super().__init__(config, config.paths.raw_weather)

    def collect(
        self,
        *,
        latitude: float,
        longitude: float,
        start: str,
        end: str,
        **_: Any,
    ) -> list[RawAsset]:
        """Fetch daily data for one point and store the raw JSON response.

        start/end: YYYYMMDD strings.
        """
        params = {
            "parameters": ",".join(_PARAMETERS),
            "community": "AG",
            "latitude": latitude,
            "longitude": longitude,
            "start": start,
            "end": end,
            "format": "JSON",
        }
        url = f"{_BASE_URL}?{httpx.QueryParams(params)}"
        dest_dir = self.raw_dir / "NASA_POWER"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"power_{latitude:.2f}_{longitude:.2f}_{start}_{end}.json"

        transport = httpx.HTTPTransport(retries=3)
        try:
            with httpx.Client(transport=transport, timeout=180.0) as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise CollectionError(f"NASA POWER request failed for ({latitude}, {longitude}): {exc}") from exc

        if "properties" not in payload or "parameter" not in payload.get("properties", {}):
            raise CollectionError(f"Unexpected NASA POWER response shape for ({latitude}, {longitude})")

        dest.write_text(json.dumps(payload), encoding="utf-8")
        asset = self._register(
            dest,
            source="NASA_POWER",
            product="POWER_DAILY_POINT",
            extra={"latitude": latitude, "longitude": longitude, "start": start, "end": end},
        )
        logger.info("[weather] NASA POWER point (%.2f, %.2f) %s–%s stored", latitude, longitude, start, end)
        return [asset]

    @staticmethod
    def to_dataframe(raw_json_path: str) -> pd.DataFrame:
        """Parse a stored POWER response into canonical pipeline columns."""
        payload = json.loads(open(raw_json_path, encoding="utf-8").read())
        lon, lat = payload["geometry"]["coordinates"][:2]
        series = payload["properties"]["parameter"]

        frame = pd.DataFrame(
            {canonical: pd.Series(series[power_name]) for power_name, canonical in _PARAMETERS.items()}
        )
        frame.index.name = "date"
        frame = frame.reset_index()
        frame["timestamp"] = pd.to_datetime(frame["date"], format="%Y%m%d", utc=True)
        frame = frame.drop(columns=["date"])
        frame["latitude"] = lat
        frame["longitude"] = lon
        frame = frame.replace(_FILL_VALUE, np.nan)
        frame["pressure_hpa"] = frame["pressure_hpa"] * 10.0  # kPa → hPa
        return frame

import json

import pytest

from data_pipeline.collectors.base_collector import CollectionError
from data_pipeline.collectors.rainfall_collector import RainfallCollector
from data_pipeline.collectors.satellite_collector import SatelliteCollector
from data_pipeline.collectors.weather_collector import WeatherCollector


def test_satellite_collector_ingests_order_directory(config, image_dir):
    collector = SatelliteCollector(config)
    assets = collector.collect(order_dir=image_dir, product="INSAT-3D_IMG_TIR1")

    # All three .png files ingested unchanged (corruption is a preprocessing
    # concern; raw acquisition preserves the delivery exactly).
    assert len(assets) == 3
    for asset in assets:
        assert asset.source == "MOSDAC"
        assert len(asset.sha256) == 64
        assert asset.size_bytes > 0

    # Catalog is persisted as JSON lines and re-readable.
    cataloged = collector.catalog()
    assert len(cataloged) == 3
    with collector.catalog_path.open() as fh:
        first = json.loads(fh.readline())
    assert first["product"] == "INSAT-3D_IMG_TIR1"


def test_collectors_refuse_to_run_without_a_real_source(config):
    with pytest.raises(CollectionError):
        SatelliteCollector(config).collect()
    with pytest.raises(CollectionError):
        RainfallCollector(config).collect()
    with pytest.raises(CollectionError):
        WeatherCollector(config).collect()


def test_ingest_missing_file_raises(config):
    with pytest.raises(CollectionError):
        RainfallCollector(config).ingest_local_file(
            "/nonexistent/rain.csv", source="IMD", product="IMD_GRIDDED_DAILY"
        )

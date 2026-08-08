import numpy as np

from data_pipeline.preprocessors.satellite_processor import SatelliteProcessor


def test_preprocess_image_is_tensor_ready(config, image_dir):
    processor = SatelliteProcessor(config)
    array = processor.preprocess_image(image_dir / "scene_a.png")

    size = config.satellite.image_size
    assert array.shape == (size, size, 1)  # grayscale → explicit channel dim
    assert array.dtype == np.float32
    assert 0.0 <= array.min() and array.max() <= 1.0

    chw = processor.to_tensor_layout(array)
    assert chw.shape == (1, size, size)


def test_corrupted_file_detection(config, image_dir):
    processor = SatelliteProcessor(config)
    assert processor.is_corrupted(image_dir / "broken.png") is True
    assert processor.is_corrupted(image_dir / "scene_a.png") is False


def test_process_directory_skips_corrupted_and_indexes_output(config, image_dir):
    processor = SatelliteProcessor(config)
    summary = processor.process_directory(image_dir, product="TEST_PRODUCT")

    assert summary.processed == 2
    assert len(summary.corrupted) == 1
    assert summary.index_path is not None and summary.index_path.exists()

    outputs = list((config.paths.processed_images / "TEST_PRODUCT").glob("*.npy"))
    assert len(outputs) == 2
    loaded = np.load(outputs[0])
    assert loaded.shape[0] == config.satellite.image_size

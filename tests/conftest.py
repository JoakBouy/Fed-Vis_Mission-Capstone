"""Pytest configuration and fixtures for Fed-Vis tests.

This module provides shared fixtures for testing, including:
- Sample configuration objects
- Synthetic test data
- Temporary directories for test outputs
"""

from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def sample_config() -> DictConfig:
    """Create a sample Hydra configuration for testing.

    Returns:
        DictConfig with test-appropriate settings.
    """
    config = {
        "seed": 42,
        "device": "cpu",
        "paths": {
            "data_root": "/tmp/fedvis_test_data",
            "output_dir": "/tmp/fedvis_test_output",
            "checkpoint_dir": "/tmp/fedvis_test_output/checkpoints",
            "log_dir": "/tmp/fedvis_test_output/logs",
        },
        "data": {
            "name": "test",
            "processed_path": "/tmp/fedvis_test_data/processed",
            "volume_size": {"depth": 32, "height": 64, "width": 64},
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "sites": ["site1", "site2"],
            "use_modality": "T1",
            "binary_labels": True,
            "preprocessing": {
                "intensity_normalization": "zscore",
            },
        },
        "model": {
            "name": "attention_unet",
            "in_channels": 1,
            "out_channels": 1,
            "base_filters": 16,
            "depth": 3,
            "norm_type": "instance",
            "dropout_rate": 0.0,
            "attention": {"enabled": True},
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "epochs": 10,
            "loss": {"name": "dice_bce", "dice_weight": 1.0, "bce_weight": 1.0},
        },
    }
    return OmegaConf.create(config)


@pytest.fixture
def synthetic_volume() -> np.ndarray:
    """Create a synthetic 3D volume for testing.

    Returns:
        Random volume of shape (32, 64, 64).
    """
    rng = np.random.default_rng(42)
    return rng.standard_normal((32, 64, 64)).astype(np.float32)


@pytest.fixture
def synthetic_label() -> np.ndarray:
    """Create a synthetic binary segmentation label.

    Returns:
        Binary label of shape (32, 64, 64) with a spherical "lesion".
    """
    label = np.zeros((32, 64, 64), dtype=np.int64)

    # Create a spherical region in the center
    center = np.array([16, 32, 32])
    radius = 10

    for z in range(32):
        for y in range(64):
            for x in range(64):
                if np.linalg.norm(np.array([z, y, x]) - center) < radius:
                    label[z, y, x] = 1

    return label


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory with mock dataset structure.

    Args:
        tmp_path: Pytest's built-in temporary path fixture.

    Yields:
        Path to temporary data directory.
    """
    # Create mock dataset structure
    data_dir = tmp_path / "data"
    site_dir = data_dir / "site1" / "patient001"
    site_dir.mkdir(parents=True)

    # We don't create actual NIfTI files here since that requires nibabel
    # Tests using real NIfTI should use separate integration tests

    yield data_dir

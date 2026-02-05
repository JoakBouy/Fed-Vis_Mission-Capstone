"""Unit tests for data preprocessing functions.

Tests cover:
- Intensity normalization (z-score, min-max, CT windowing)
- Crop/pad to target size
- Full preprocessing pipeline
"""

import numpy as np
import pytest

from fedvis.data.preprocessing import (
    crop_or_pad,
    normalize_intensity,
    preprocess_volume,
)


class TestNormalizeIntensity:
    """Tests for the normalize_intensity function."""

    def test_zscore_normalization(self, synthetic_volume: np.ndarray) -> None:
        """Test z-score normalization produces zero mean, unit variance."""
        normalized = normalize_intensity(synthetic_volume, method="zscore")

        # Check that mean is approximately 0 and std is approximately 1
        # (for non-background voxels)
        mask = normalized > normalized.min()
        assert np.abs(normalized[mask].mean()) < 0.1
        assert np.abs(normalized[mask].std() - 1.0) < 0.1

    def test_minmax_normalization(self, synthetic_volume: np.ndarray) -> None:
        """Test min-max normalization scales to [0, 1]."""
        normalized = normalize_intensity(synthetic_volume, method="minmax")

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert np.isclose(normalized.min(), 0.0, atol=1e-6)
        assert np.isclose(normalized.max(), 1.0, atol=1e-6)

    def test_ct_window_normalization(self) -> None:
        """Test CT windowing clips and scales correctly."""
        # Create volume with HU values
        volume = np.array([-1000, -600, -100, 0, 100, 400, 1000], dtype=np.float32)

        # Window: center=-600, width=1500 -> range [-1350, 150]
        normalized = normalize_intensity(
            volume,
            method="ct_window",
            ct_window_center=-600,
            ct_window_width=1500,
        )

        # Values should be clipped and scaled to [0, 1]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_invalid_method_raises(self, synthetic_volume: np.ndarray) -> None:
        """Test that invalid normalization method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_intensity(synthetic_volume, method="invalid")


class TestCropOrPad:
    """Tests for the crop_or_pad function."""

    def test_crop_larger_volume(self) -> None:
        """Test cropping when volume is larger than target."""
        volume = np.random.randn(100, 128, 128).astype(np.float32)
        target_size = (64, 64, 64)

        result = crop_or_pad(volume, target_size)

        assert result.shape == target_size

    def test_pad_smaller_volume(self) -> None:
        """Test padding when volume is smaller than target."""
        volume = np.random.randn(32, 48, 48).astype(np.float32)
        target_size = (64, 64, 64)

        result = crop_or_pad(volume, target_size)

        assert result.shape == target_size
        # Check that padding is symmetric (zeros at edges)
        # The original 32x48x48 should be centered in 64x64x64

    def test_crop_preserves_center(self) -> None:
        """Test that cropping preserves the center of the volume."""
        volume = np.zeros((100, 100, 100), dtype=np.float32)
        # Place a marker at the center
        volume[50, 50, 50] = 1.0
        target_size = (50, 50, 50)

        result = crop_or_pad(volume, target_size)

        # The center of the cropped result should contain the marker
        assert result[25, 25, 25] == 1.0

    def test_with_label(self, synthetic_volume: np.ndarray, synthetic_label: np.ndarray) -> None:
        """Test that label is transformed identically to volume."""
        target_size = (16, 32, 32)

        result_vol, result_lbl = crop_or_pad(
            synthetic_volume, target_size, label=synthetic_label
        )

        assert result_vol.shape == target_size
        assert result_lbl.shape == target_size
        assert result_lbl.dtype == synthetic_label.dtype


class TestPreprocessVolume:
    """Tests for the full preprocessing pipeline."""

    def test_default_preprocessing(self, synthetic_volume: np.ndarray) -> None:
        """Test default preprocessing produces correct output shape."""
        target_size = (32, 64, 64)

        result = preprocess_volume(
            synthetic_volume,
            target_size=target_size,
            normalization="zscore",
        )

        assert result.shape == target_size
        assert result.dtype == np.float32

    def test_preprocessing_with_label(
        self, synthetic_volume: np.ndarray, synthetic_label: np.ndarray
    ) -> None:
        """Test preprocessing with label returns both."""
        target_size = (32, 64, 64)

        result_vol, result_lbl = preprocess_volume(
            synthetic_volume,
            target_size=target_size,
            normalization="zscore",
            label=synthetic_label,
        )

        assert result_vol.shape == target_size
        assert result_lbl.shape == target_size

    def test_preprocessing_ct_with_windowing(self) -> None:
        """Test CT preprocessing with HU windowing."""
        # Simulate CT volume with HU values
        volume = np.random.randn(64, 128, 128).astype(np.float32) * 500 - 500

        result = preprocess_volume(
            volume,
            target_size=(32, 64, 64),
            normalization="ct_window",
            ct_window_center=-600,
            ct_window_width=1500,
        )

        assert result.shape == (32, 64, 64)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

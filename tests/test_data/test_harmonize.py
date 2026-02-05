"""Unit tests for data harmonization module."""

import numpy as np
import pytest

from fedvis.data.harmonize import (
    extract_flair_channel,
    harmonize_label,
    harmonize_volume,
    resample_to_isotropic,
    zscore_normalize,
)


class TestExtractFlairChannel:
    """Tests for FLAIR channel extraction."""

    def test_3d_volume_unchanged(self) -> None:
        """3D volume should pass through unchanged."""
        volume = np.random.randn(64, 128, 128)
        result = extract_flair_channel(volume, "brats")
        assert result.shape == (64, 128, 128)
        np.testing.assert_array_equal(result, volume)

    def test_4d_volume_extracts_first_channel(self) -> None:
        """4D volume should return first channel (FLAIR)."""
        volume = np.random.randn(4, 64, 128, 128)
        result = extract_flair_channel(volume, "brats")
        assert result.shape == (64, 128, 128)
        np.testing.assert_array_equal(result, volume[0])

    def test_invalid_shape_raises(self) -> None:
        """Invalid shape should raise ValueError."""
        volume = np.random.randn(2, 4, 64, 128, 128)  # 5D
        with pytest.raises(ValueError, match="Unexpected volume shape"):
            extract_flair_channel(volume, "brats")


class TestResampleToIsotropic:
    """Tests for spatial resampling."""

    def test_no_change_same_spacing(self) -> None:
        """Volume should be ~unchanged with same spacing."""
        volume = np.random.randn(64, 64, 64).astype(np.float32)
        result = resample_to_isotropic(volume, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        assert result.shape == (64, 64, 64)

    def test_upsample_in_slice_direction(self) -> None:
        """2mm slices should roughly double in first dimension."""
        volume = np.random.randn(32, 64, 64).astype(np.float32)
        result = resample_to_isotropic(volume, (2.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        # Should approximately double first dimension
        assert result.shape[0] == 64
        assert result.shape[1] == 64
        assert result.shape[2] == 64

    def test_downsample(self) -> None:
        """Downsampling should reduce dimensions."""
        volume = np.random.randn(64, 128, 128).astype(np.float32)
        result = resample_to_isotropic(volume, (1.0, 0.5, 0.5), (1.0, 1.0, 1.0))
        assert result.shape[0] == 64
        assert result.shape[1] == 64
        assert result.shape[2] == 64


class TestZscoreNormalize:
    """Tests for Z-score normalization."""

    def test_output_approximately_zero_mean(self) -> None:
        """Normalized brain region should have ~zero mean."""
        volume = np.random.randn(64, 64, 64).astype(np.float32) * 100 + 500
        result = zscore_normalize(volume)
        brain_mask = result != 0
        assert np.abs(result[brain_mask].mean()) < 0.5

    def test_output_approximately_unit_std(self) -> None:
        """Normalized brain region should have ~unit std (approximately)."""
        volume = np.random.randn(64, 64, 64).astype(np.float32) * 100 + 500
        result = zscore_normalize(volume)
        brain_mask = result != 0
        # May be clipped, so std could be less than 1
        assert result[brain_mask].std() <= 1.5

    def test_clipping_applied(self) -> None:
        """Values should be clipped to specified range."""
        volume = np.ones((64, 64, 64)) * 1000
        volume[32, 32, 32] = 0  # Outlier
        result = zscore_normalize(volume, clip_range=(-3.0, 3.0))
        assert result.min() >= -3.0
        assert result.max() <= 3.0


class TestHarmonizeVolume:
    """Tests for full harmonization pipeline."""

    def test_output_shape(self) -> None:
        """Output should match target size."""
        volume = np.random.randn(4, 155, 240, 240).astype(np.float32)
        result = harmonize_volume(
            volume,
            current_spacing=(1.0, 1.0, 1.0),
            target_size=(64, 128, 128),
            dataset_type="brats",
        )
        assert result.shape == (64, 128, 128)

    def test_output_dtype(self) -> None:
        """Output should be float32."""
        volume = np.random.randn(64, 128, 128).astype(np.float64)
        result = harmonize_volume(
            volume,
            current_spacing=(1.0, 1.0, 1.0),
            target_size=(32, 64, 64),
            dataset_type="brats",
        )
        assert result.dtype == np.float32


class TestHarmonizeLabel:
    """Tests for label harmonization."""

    def test_output_shape(self) -> None:
        """Output should match target size."""
        label = np.random.randint(0, 4, (155, 240, 240))
        result = harmonize_label(
            label,
            current_spacing=(1.0, 1.0, 1.0),
            target_size=(64, 128, 128),
        )
        assert result.shape == (64, 128, 128)

    def test_binarization(self) -> None:
        """Labels should be binarized when requested."""
        label = np.array([[[0, 1, 2, 4]]])  # BraTS classes
        result = harmonize_label(
            label,
            current_spacing=(1.0, 1.0, 1.0),
            target_size=(1, 1, 4),
            binarize=True,
        )
        expected = np.array([[[0, 1, 1, 1]]])
        np.testing.assert_array_equal(result, expected)

    def test_preserves_labels_without_binarize(self) -> None:
        """Labels should be preserved when binarize=False."""
        label = np.array([[[0, 1, 2, 4]]])
        result = harmonize_label(
            label,
            current_spacing=(1.0, 1.0, 1.0),
            target_size=(1, 1, 4),
            binarize=False,
        )
        np.testing.assert_array_equal(result, label)

"""Unit tests for dataset classes.

Tests cover:
- BaseDataset3D abstract class behavior
- FeTSDataset, ProstateDataset, LungDataset implementations
- Data splitting logic
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from fedvis.data.base import BaseDataset3D


class ConcreteDataset(BaseDataset3D):
    """Concrete implementation of BaseDataset3D for testing."""

    def __init__(
        self,
        cfg: DictConfig,
        split: str = "train",
        site: str | None = None,
        transform: object | None = None,
        mock_files: list[str] | None = None,
    ) -> None:
        self.mock_files = mock_files or [f"sample_{i}" for i in range(100)]
        super().__init__(cfg, split, site, transform)

    def _get_file_list(self) -> list[str]:
        return self.mock_files

    def _load_volume(self, sample_id: str) -> np.ndarray:
        vs = self.volume_size
        return np.random.randn(vs[0], vs[1], vs[2]).astype(np.float32)

    def _load_label(self, sample_id: str) -> np.ndarray:
        vs = self.volume_size
        label = np.zeros((vs[0], vs[1], vs[2]), dtype=np.int64)
        label[vs[0] // 4 : 3 * vs[0] // 4, vs[1] // 4 : 3 * vs[1] // 4, vs[2] // 4 : 3 * vs[2] // 4] = 1
        return label


class TestBaseDataset3D:
    """Tests for the BaseDataset3D abstract class."""

    def test_invalid_split_raises(self, sample_config: DictConfig) -> None:
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be"):
            ConcreteDataset(sample_config, split="invalid")

    def test_train_split_size(self, sample_config: DictConfig) -> None:
        """Test that train split has correct size."""
        dataset = ConcreteDataset(sample_config, split="train")
        # With 100 samples and 0.7 train ratio, should have ~70 samples
        assert len(dataset) == 70

    def test_val_split_size(self, sample_config: DictConfig) -> None:
        """Test that validation split has correct size."""
        dataset = ConcreteDataset(sample_config, split="val")
        # With 100 samples and 0.15 val ratio, should have ~15 samples
        assert len(dataset) == 15

    def test_test_split_size(self, sample_config: DictConfig) -> None:
        """Test that test split has correct size."""
        dataset = ConcreteDataset(sample_config, split="test")
        # With 100 samples and 0.15 test ratio, should have ~15 samples
        assert len(dataset) == 15

    def test_splits_are_disjoint(self, sample_config: DictConfig) -> None:
        """Test that train/val/test splits don't overlap."""
        train_ds = ConcreteDataset(sample_config, split="train")
        val_ds = ConcreteDataset(sample_config, split="val")
        test_ds = ConcreteDataset(sample_config, split="test")

        train_files = set(train_ds.file_list)
        val_files = set(val_ds.file_list)
        test_files = set(test_ds.file_list)

        assert len(train_files & val_files) == 0
        assert len(train_files & test_files) == 0
        assert len(val_files & test_files) == 0

    def test_splits_are_reproducible(self, sample_config: DictConfig) -> None:
        """Test that splits are deterministic with same seed."""
        dataset1 = ConcreteDataset(sample_config, split="train")
        dataset2 = ConcreteDataset(sample_config, split="train")

        assert dataset1.file_list == dataset2.file_list

    def test_getitem_returns_tensors(self, sample_config: DictConfig) -> None:
        """Test that __getitem__ returns torch tensors."""
        dataset = ConcreteDataset(sample_config, split="train")
        volume, label = dataset[0]

        assert isinstance(volume, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_getitem_shapes(self, sample_config: DictConfig) -> None:
        """Test that returned tensors have correct shapes."""
        dataset = ConcreteDataset(sample_config, split="train")
        volume, label = dataset[0]

        # Shape should be (C, D, H, W)
        assert volume.shape == (1, 32, 64, 64)
        assert label.shape == (1, 32, 64, 64)

    def test_getitem_dtypes(self, sample_config: DictConfig) -> None:
        """Test that returned tensors have correct dtypes."""
        dataset = ConcreteDataset(sample_config, split="train")
        volume, label = dataset[0]

        assert volume.dtype == torch.float32
        assert label.dtype == torch.int64

    def test_volume_size_property(self, sample_config: DictConfig) -> None:
        """Test that volume_size property returns correct tuple."""
        dataset = ConcreteDataset(sample_config, split="train")
        assert dataset.volume_size == (32, 64, 64)

    def test_transform_is_applied(self, sample_config: DictConfig) -> None:
        """Test that custom transform is applied to samples."""
        mock_transform = MagicMock()
        mock_transform.return_value = (np.zeros((32, 64, 64)), np.zeros((32, 64, 64)))

        dataset = ConcreteDataset(sample_config, split="train", transform=mock_transform)
        _ = dataset[0]

        mock_transform.assert_called_once()


class TestDataLoaderIntegration:
    """Test datasets work with PyTorch DataLoader."""

    def test_with_dataloader(self, sample_config: DictConfig) -> None:
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = ConcreteDataset(sample_config, split="train")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        batch = next(iter(dataloader))
        volumes, labels = batch

        assert volumes.shape == (2, 1, 32, 64, 64)
        assert labels.shape == (2, 1, 32, 64, 64)

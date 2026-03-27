"""Medical image dataset utilities.

Provides MedDataset (raw NIfTI loading with augmentation + normalization),
file-pair finders for BraTS/FeTS and Prostate formats, and thin dataset
wrapper classes for the three organs used in Fed-Vis.
"""

import os
import random
from glob import glob

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

# Default target volume shape (D, H, W)
TARGET = (64, 128, 128)


def crop_or_pad(vol: np.ndarray, target: tuple) -> np.ndarray:
    """Crop or zero-pad a volume to the target shape."""
    out = np.zeros(target, dtype=vol.dtype)
    slices = tuple(slice(0, min(v, t)) for v, t in zip(vol.shape, target))
    out[slices] = vol[slices]
    return out


class MedDataset(Dataset):
    """Load (image, mask) pairs from NIfTI files with preprocessing and augmentation.

    Args:
        imgs:  List of image file paths (.nii.gz).
        masks: List of corresponding mask file paths (.nii.gz).
        aug:   Whether to apply random augmentation (flips, rotation, intensity).
        norm:  Normalisation method: 'zscore' (MRI) or 'percentile' (CT).
    """

    def __init__(self, imgs: list, masks: list, aug: bool = False, norm: str = "zscore") -> None:
        assert len(imgs) == len(masks), "imgs and masks must have the same length"
        self.imgs = imgs
        self.masks = masks
        self.aug = aug
        self.norm = norm

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int):
        vol = nib.load(self.imgs[i]).get_fdata().astype(np.float32)
        msk = nib.load(self.masks[i]).get_fdata().astype(np.float32)

        # Handle 4-D files (e.g. multi-modality FeTS) — take first channel
        if vol.ndim == 4:
            vol = vol[..., 0]
        if msk.ndim == 4:
            msk = msk[..., 0]

        # Intensity normalisation
        if self.norm == "zscore":
            fg = vol > 0
            if fg.sum() > 0:
                vol[fg] = (vol[fg] - vol[fg].mean()) / (vol[fg].std() + 1e-8)
            vol[~fg] = 0.0
        else:  # percentile clip (CT / prostate)
            lo, hi = np.percentile(vol, [0.5, 99.5])
            if hi > lo:
                vol = np.clip((vol - lo) / (hi - lo), 0, 1)

        # Binarise mask
        msk = (msk > 0).astype(np.float32)

        # Resize to TARGET using zoom
        if vol.shape != TARGET:
            factors = [t / s for t, s in zip(TARGET, vol.shape)]
            vol = zoom(vol, factors, order=1)
            msk = zoom(msk, factors, order=0)

        vol = crop_or_pad(vol, TARGET)
        msk = crop_or_pad(msk, TARGET)
        msk = (msk > 0.5).astype(np.float32)

        # Augmentation
        if self.aug:
            # Spatial: random flips along each axis
            for ax in range(3):
                if random.random() > 0.5:
                    vol = np.flip(vol, ax).copy()
                    msk = np.flip(msk, ax).copy()
            # Spatial: random 90-degree rotation in the axial plane
            k = random.randint(0, 3)
            if k:
                vol = np.rot90(vol, k, (1, 2)).copy()
                msk = np.rot90(msk, k, (1, 2)).copy()
            # Intensity: Gaussian noise
            if random.random() > 0.5:
                vol = vol + np.random.normal(0, 0.03, vol.shape).astype(np.float32)
            # Intensity: brightness shift
            if random.random() > 0.5:
                vol = vol * float(np.random.uniform(0.9, 1.1))
            # Intensity: contrast shift
            if random.random() > 0.5:
                mean = vol.mean()
                vol = (vol - mean) * float(np.random.uniform(0.85, 1.15)) + mean

        return (
            torch.from_numpy(vol.copy()).unsqueeze(0).float(),
            torch.from_numpy(msk.copy()).unsqueeze(0).float(),
        )


# ---------------------------------------------------------------------------
# File-pair finders
# ---------------------------------------------------------------------------

def find_brats(d: str):
    """Return (imgs, masks) lists for a BraTS / FeTS directory tree.

    Looks for files matching ``*seg*.nii.gz`` as masks and picks the first
    non-seg sibling as the corresponding image.
    """
    imgs, masks = [], []
    for seg in sorted(glob(os.path.join(d, "**/*seg*.nii.gz"), recursive=True)):
        bn = os.path.basename(seg)
        if bn.startswith("._"):
            continue
        cands = [
            c for c in glob(os.path.join(os.path.dirname(seg), "*.nii.gz"))
            if "seg" not in c.lower() and not os.path.basename(c).startswith("._")
        ]
        if cands:
            imgs.append(sorted(cands)[0])
            masks.append(seg)
    return imgs, masks


def find_prostate(d: str):
    """Return (imgs, masks) lists for a Prostate MRI directory tree.

    First tries the BraTS convention (``*seg*``). Falls back to looking for
    ``<name>_segmentation.nii.gz`` alongside ``<name>.nii.gz``.
    """
    imgs, masks = find_brats(d)
    if imgs:
        return imgs, masks

    imgs, masks = [], []
    for f in sorted(glob(os.path.join(d, "**/*.nii.gz"), recursive=True)):
        if "seg" in f.lower() or os.path.basename(f).startswith("._"):
            continue
        seg_path = f.replace(".nii.gz", "_segmentation.nii.gz")
        if os.path.exists(seg_path):
            imgs.append(f)
            masks.append(seg_path)
    return imgs, masks


# ---------------------------------------------------------------------------
# Shared split helper
# ---------------------------------------------------------------------------

def _split_pairs(
    imgs: list,
    masks: list,
    train_ratio: float = 0.85,
    seed: int = 42,
    max_samples: int = None,
):
    """Shuffle, optionally cap, then split into train/val."""
    n = len(imgs)
    if max_samples:
        n = min(n, max_samples)
    ii = list(range(len(imgs)))
    rng = random.Random(seed)
    rng.shuffle(ii)
    ii = ii[:n]
    s = int(train_ratio * n)
    train_idx, val_idx = ii[:s], ii[s:]
    return (
        [imgs[i] for i in train_idx],
        [masks[i] for i in train_idx],
        [imgs[i] for i in val_idx],
        [masks[i] for i in val_idx],
    )


# ---------------------------------------------------------------------------
# Per-organ wrapper classes (used by train_local.py / build_loaders)
# ---------------------------------------------------------------------------

class FeTSDataset(MedDataset):
    """Brain-tumour segmentation dataset (FeTS / BraTS format).

    Args:
        cfg:   OmegaConf config with ``cfg.data.processed_path``.
        split: 'train' or 'val'.
        site:  Optional sub-directory name (hospital site).
    """

    def __init__(self, cfg, split: str = "train", site: str = None) -> None:
        root = str(cfg.data.processed_path)
        search_dir = os.path.join(root, site) if site and os.path.isdir(os.path.join(root, site)) else root
        imgs, masks = find_brats(search_dir)
        tr_i, tr_m, val_i, val_m = _split_pairs(imgs, masks, seed=42)
        if split == "train":
            super().__init__(tr_i, tr_m, aug=True, norm="zscore")
        else:
            super().__init__(val_i, val_m, aug=False, norm="zscore")


class ProstateDataset(MedDataset):
    """Prostate MRI segmentation dataset wrapper.

    Args:
        cfg:   OmegaConf config with ``cfg.data.processed_path``.
        split: 'train' or 'val'.
        site:  Optional sub-directory name (hospital site).
    """

    def __init__(self, cfg, split: str = "train", site: str = None) -> None:
        root = str(cfg.data.processed_path)
        search_dir = os.path.join(root, site) if site and os.path.isdir(os.path.join(root, site)) else root
        imgs, masks = find_prostate(search_dir)
        tr_i, tr_m, val_i, val_m = _split_pairs(imgs, masks, seed=42)
        if split == "train":
            super().__init__(tr_i, tr_m, aug=True, norm="percentile")
        else:
            super().__init__(val_i, val_m, aug=False, norm="percentile")


class LungDataset(MedDataset):
    """CT Lung nodule segmentation dataset wrapper.

    Args:
        cfg:   OmegaConf config with ``cfg.data.processed_path``.
        split: 'train' or 'val'.
        site:  Optional sub-directory name (hospital site).
    """

    def __init__(self, cfg, split: str = "train", site: str = None) -> None:
        root = str(cfg.data.processed_path)
        search_dir = os.path.join(root, site) if site and os.path.isdir(os.path.join(root, site)) else root
        imgs, masks = find_brats(search_dir)
        tr_i, tr_m, val_i, val_m = _split_pairs(imgs, masks, seed=42)
        if split == "train":
            super().__init__(tr_i, tr_m, aug=True, norm="percentile")
        else:
            super().__init__(val_i, val_m, aug=False, norm="percentile")


__all__ = [
    "TARGET",
    "crop_or_pad",
    "MedDataset",
    "find_brats",
    "find_prostate",
    "_split_pairs",
    "FeTSDataset",
    "ProstateDataset",
    "LungDataset",
]

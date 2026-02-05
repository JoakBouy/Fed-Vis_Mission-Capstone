# Data Pipeline Documentation

This document describes the data loading and preprocessing pipeline for Fed-Vis.

## Supported Datasets

| Dataset | Modality | Anatomy | Source |
|---------|----------|---------|--------|
| FeTS 2022 | MRI (T1, T1ce, T2, FLAIR) | Brain Tumor | [FeTS Challenge](https://www.synapse.org/fets) |
| Prostate | MRI (T2) | Prostate Gland | Multi-site collection |
| Lung CT | CT | Lung Nodules | CT Lung dataset |

## Data Format

All datasets are converted to a unified format:

```
Tensor Shape: (C, D, H, W)
- C: Channels (1 for single modality)
- D: Depth (64 slices)
- H: Height (128 pixels)
- W: Width (128 pixels)

Volume dtype: float32 (normalized)
Label dtype: int64 (class indices)
```

## Preprocessing Pipeline

```mermaid
flowchart LR
    A[NIfTI File] --> B[Load Volume]
    B --> C[Resample Spacing]
    C --> D[Normalize Intensity]
    D --> E[Crop/Pad to 64×128×128]
    E --> F[Tensor Output]
```

### 1. Loading

NIfTI files are loaded using `nibabel` and converted to numpy arrays.

```python
from fedvis.data.preprocessing import load_nifti

volume, spacing = load_nifti("brain_t1.nii.gz")
# volume: np.ndarray (D, H, W)
# spacing: np.ndarray [spacing_d, spacing_h, spacing_w] in mm
```

### 2. Intensity Normalization

Three methods available:

| Method | Use Case | Formula |
|--------|----------|---------|
| `zscore` | MRI | `(x - μ) / σ` on non-background voxels |
| `minmax` | General | `(x - min) / (max - min)` |
| `ct_window` | CT | Clip to HU window, then scale to [0, 1] |

```python
from fedvis.data.preprocessing import normalize_intensity

# For MRI
normalized = normalize_intensity(volume, method="zscore")

# For CT with lung window
normalized = normalize_intensity(
    volume,
    method="ct_window",
    ct_window_center=-600,
    ct_window_width=1500
)
```

### 3. Spatial Normalization

Volumes are resampled to target spacing and cropped/padded to target size:

```python
from fedvis.data.preprocessing import preprocess_volume

processed = preprocess_volume(
    volume,
    target_size=(64, 128, 128),
    normalization="zscore",
    current_spacing=spacing,
    target_spacing=np.array([1.0, 1.0, 1.0])
)
```

## Configuration

Dataset parameters are configured via Hydra YAML files:

```yaml
# configs/data/fets.yaml
volume_size:
  depth: 64
  height: 128
  width: 128

preprocessing:
  intensity_normalization: zscore
```

Override via CLI:
```bash
python -m fedvis.scripts.train_local data=prostate data.volume_size.depth=32
```

## Usage Example

```python
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from fedvis.data import FeTSDataset

# Load config
cfg = OmegaConf.load("configs/config.yaml")

# Create dataset
dataset = FeTSDataset(cfg, split="train", site="1")

# Use with DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for volumes, labels in loader:
    print(f"Batch: {volumes.shape}, {labels.shape}")
    # Batch: torch.Size([2, 1, 64, 128, 128])
```

## Testing

Run data pipeline tests:

```bash
poetry run pytest tests/test_data/ -v
```

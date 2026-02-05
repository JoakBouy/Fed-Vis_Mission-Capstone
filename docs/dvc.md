# Data Version Control (DVC) Setup

## Installation

DVC is included in the project dependencies. After running `poetry install`, you can use DVC:

```bash
poetry run dvc --version
```

If DVC is not available, install it separately:

```bash
pip install dvc
# Or for remote storage support:
pip install dvc[s3]   # AWS S3
pip install dvc[gs]   # Google Cloud
pip install dvc[azure] # Azure Blob
```

## Initializing DVC (First Time)

If starting fresh:
```bash
cd fed_vis
dvc init
```

## Adding Data

```bash
# Add a data directory
dvc add data/FeTS2022

# This creates data/FeTS2022.dvc - commit this file
git add data/FeTS2022.dvc data/.gitignore
git commit -m "Add FeTS2022 dataset"
```

## Configuring Remote Storage

### Local Storage (Development)
```bash
dvc remote add -d local /path/to/dvc_cache
```

### AWS S3
```bash
dvc remote add -d s3remote s3://your-bucket/dvc-storage
dvc remote modify s3remote access_key_id YOUR_KEY
dvc remote modify s3remote secret_access_key YOUR_SECRET
```

### Google Cloud Storage
```bash
dvc remote add -d gcs gs://your-bucket/dvc-storage
```

## Pushing/Pulling Data

```bash
# Push data to remote
dvc push

# Pull data from remote
dvc pull
```

## Workflow

1. **Add new data**: `dvc add data/new_dataset`
2. **Commit .dvc file**: `git add data/new_dataset.dvc && git commit`
3. **Push data**: `dvc push`
4. **Clone repo**: `git clone ...`
5. **Get data**: `dvc pull`

# Hydra Configuration Files

This directory contains all configuration files for Fed-Vis, managed by [Hydra](https://hydra.cc/).

## Structure

```
configs/
├── config.yaml           # Main entry point (composes other configs)
├── data/                 # Dataset configurations
│   ├── fets.yaml         # FeTS 2022 Brain MRI
│   ├── prostate.yaml     # Multi-site Prostate MRI
│   └── lung.yaml         # Lung CT
├── model/                # Model architectures
│   └── attention_unet.yaml
├── training/             # Training hyperparameters
│   └── default.yaml
└── federation/           # Federated learning settings
    └── default.yaml
```

## Usage

### View Full Configuration
```bash
python -m fedvis.scripts.train_local --cfg job
```

### Override Parameters via CLI
```bash
# Change dataset
python -m fedvis.scripts.train_local data=prostate

# Override specific values
python -m fedvis.scripts.train_local training.epochs=50 training.batch_size=4

# Multiple overrides
python -m fedvis.scripts.train_local \
    data=lung \
    model.base_filters=64 \
    training.learning_rate=0.001
```

### Environment Variables

Set these environment variables to configure paths:

```bash
export FEDVIS_DATA_ROOT=/path/to/datasets
export FEDVIS_OUTPUT_DIR=/path/to/outputs
```

## Adding New Configurations

1. Create a new YAML file in the appropriate subdirectory
2. Add it to the defaults list in `config.yaml` if it should be loaded by default
3. Or select it via CLI: `data=my_new_dataset`

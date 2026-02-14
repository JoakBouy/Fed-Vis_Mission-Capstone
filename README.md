# Fed-Vis

Privacy-Preserving 3D Medical Image Segmentation with Federated Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade implementation of federated learning for 3D medical image segmentation, designed for privacy-preserving collaborative AI in healthcare.

## Key Features

- **Attention U-Net**: Custom 3D segmentation model with visualizable attention gates
- **Federated Learning**: Flower-based FL simulation with multiple client nodes
- **3D Visualization**: Export to NIfTI, OBJ mesh, and JSON for Three.js rendering
- **Production Ready**: Poetry, Hydra configs, DVC data versioning, FastAPI inference

## Project Structure

```
fed_vis/
├── configs/              # Hydra configuration files
│   ├── config.yaml
│   ├── data/
│   └── training/
├── src/fedvis/           # Main package
│   ├── data/             # Data loaders and harmonization
│   ├── models/           # Attention U-Net architecture
│   ├── api/              # FastAPI inference service
│   └── training/         # Training pipeline
├── notebooks/            # Demo notebooks
├── tests/                # Unit tests
├── docs/                 # Documentation
└── pyproject.toml        # Dependencies
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/JoakBouy/fed-vis_Mission-Capstone.git
cd fed-vis

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
$env:PYTHONPATH="src"
python -m pytest tests/ -v
```

### Run the Demo Notebook

```bash
pip install jupyter
jupyter notebook notebooks/demo_fedvis.ipynb
```

### Run the API Server

```bash
uvicorn fedvis.api.app:app --reload --port 8000
```

Open http://localhost:8000/docs for Swagger UI.

## Model Architecture

The 3D Attention U-Net extends standard U-Net with attention gates for interpretable segmentation:

```
Input (1, 64, 128, 128)
    |
Encoder: 4 levels (64 -> 128 -> 256 -> 512)
    |
Bottleneck: 1024 channels
    |
Decoder: 4 levels with Attention Gates
    |
Output (1, 64, 128, 128)
```

**Parameters:** ~90 million trainable

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/model/info` | GET | Model architecture details |
| `/predict` | POST | Run segmentation on uploaded volume |
| `/docs` | GET | Swagger UI documentation |

## Docker Deployment

```bash
docker build -t fed-vis:latest .
docker run -p 8000:8000 fed-vis:latest
```

## References

- Oktay et al. (2018) - Attention U-Net: Learning Where to Look for the Pancreas
- McMahan et al. (2017) - Communication-Efficient Learning of Deep Networks from Decentralized Data
- Pati et al. (2022) - Federated learning enables big data for rare cancer boundary detection

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Joak Buoy Gai**  
BSc. Software Engineering  
African Leadership University

Supervisor: Dirac Murairi

## Recording

[Link to Demo Recording
]([url](https://drive.google.com/file/d/1dFa056kh6jHA9U0cJ6bqLMV3pN7Jd0ru/view?usp=sharing))

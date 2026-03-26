"""Run a federated training simulation with 2 hospital nodes.

Nodes:
  - KNH-Brain    : FeTS brain-tumour MRI  (up to 200 samples)
  - AKU-Prostate : Prostate MRI           (up to 120 samples)

Algorithm: FedProx + GroupNorm + FedAvg aggregation
Scheduler: linear warmup for the first WARMUP rounds, then cosine decay.

Usage:
    python -m fedvis.scripts.train_federated --data_root /path/to/data
    python -m fedvis.scripts.train_federated --rounds 45 --local_epochs 5 --mu 0.01
"""

import argparse
import logging
import math
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import ndarrays_to_parameters

from fedvis.models.attention_unet import AttentionUNet3D
from fedvis.federation.client import FedVisClient
from fedvis.data import MedDataset, find_brats, find_prostate, _split_pairs

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------

NODES = [
    {
        "name": "KNH-Brain",
        "dataset": "fets",
        "subdir": "FeTS2022",
        "norm": "zscore",
        "max_samples": 200,
    },
    {
        "name": "AKU-Prostate",
        "dataset": "prostate",
        "subdir": "Prostate",
        "norm": "percentile",
        "max_samples": 120,
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fed-Vis federated simulation (FedProx)")
    p.add_argument("--rounds",        type=int,   default=45)
    p.add_argument("--local_epochs",  type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--mu",            type=float, default=0.01,
                   help="FedProx proximal strength (0 = FedAvg)")
    p.add_argument("--warmup",        type=int,   default=3,
                   help="Number of warmup rounds for LR schedule")
    p.add_argument("--features",      type=int,   default=32,
                   help="Base channel count (AttentionUNet3D base_filters)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output",        type=str,   default="outputs/federated")
    p.add_argument("--data_root",     type=str,   default="data",
                   help="Root directory containing FeTS2022/ and Prostate/ subdirs")
    p.add_argument("--cpus",          type=int,   default=2)
    p.add_argument("--gpus",          type=float, default=0.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def compute_lr(rnd_0indexed: int, base_lr: float, warmup: int, total_rounds: int) -> float:
    """Linear warmup then cosine decay."""
    if rnd_0indexed < warmup:
        return base_lr * (rnd_0indexed + 1) / warmup
    return base_lr * 0.5 * (
        1 + math.cos(math.pi * (rnd_0indexed - warmup) / max(total_rounds - warmup, 1))
    )


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def make_client_fn(nodes, args, device):
    """Factory called by Flower to create each simulated client."""
    # Pre-build loaders so each client always gets the same split
    node_loaders = {}
    for node in nodes:
        data_dir = os.path.join(args.data_root, node["subdir"])
        if not os.path.isdir(data_dir):
            logger.warning(f"Data dir not found: {data_dir} — using data_root directly")
            data_dir = args.data_root

        if node["dataset"] == "fets":
            imgs, masks = find_brats(data_dir)
        else:
            imgs, masks = find_prostate(data_dir)

        if not imgs:
            logger.warning(f"[{node['name']}] No pairs found in {data_dir}")

        tr_i, tr_m, val_i, val_m = _split_pairs(
            imgs, masks, max_samples=node["max_samples"], seed=args.seed
        )

        train_ds = MedDataset(tr_i, tr_m, aug=True, norm=node["norm"])
        val_ds = MedDataset(val_i, val_m, aug=False, norm=node["norm"])

        node_loaders[node["name"]] = (
            DataLoader(train_ds, args.batch_size, shuffle=True,
                       num_workers=2, pin_memory=True),
            DataLoader(val_ds, args.batch_size, shuffle=False,
                       num_workers=2, pin_memory=True),
        )
        logger.info(
            f"[{node['name']}] {len(train_ds)} train / {len(val_ds)} val  "
            f"(norm={node['norm']})"
        )

    def client_fn(cid):
        node = nodes[int(cid)]
        model = AttentionUNet3D(1, 1, args.features)
        train_loader, val_loader = node_loaders[node["name"]]
        return FedVisClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            name=node["name"],
            cfg={"mu": args.mu},
            device=device,
        ).to_client()

    return client_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("Fed-Vis Federated Simulation  (FedProx + GroupNorm)")
    logger.info(f"  rounds={args.rounds}  local_epochs={args.local_epochs}")
    logger.info(f"  lr={args.lr}  warmup={args.warmup}  mu={args.mu}")
    logger.info(f"  nodes: {[n['name'] for n in NODES]}")

    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  device: {device}")

    # Initial global model
    init_model = AttentionUNet3D(1, 1, args.features)
    n_params = sum(p.numel() for p in init_model.parameters())
    logger.info(f"  model params: {n_params:,}")
    init_weights = [v.cpu().numpy() for _, v in init_model.state_dict().items()]

    # LR schedule passed to clients each round (Flower rounds are 1-indexed)
    def on_fit_config(rnd: int):
        lr = compute_lr(rnd - 1, args.lr, args.warmup, args.rounds)
        return {"local_epochs": args.local_epochs, "lr": lr}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(NODES),
        min_evaluate_clients=len(NODES),
        min_available_clients=len(NODES),
        initial_parameters=ndarrays_to_parameters(init_weights),
        on_fit_config_fn=on_fit_config,
    )

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(NODES, args, device),
        num_clients=len(NODES),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": args.cpus, "num_gpus": args.gpus},
    )

    # Print per-round dice
    logger.info("\n=== Results ===")
    for entry in history.metrics_distributed.get("dice", []):
        rnd, metrics = entry
        logger.info(f"  round {rnd}: {metrics}")

    if history.losses_distributed:
        logger.info(f"  final loss: {history.losses_distributed[-1]}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

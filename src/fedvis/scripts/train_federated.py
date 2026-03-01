"""Run a federated training simulation with 3 hospital nodes.

Each node holds data from a different organ:
  - node 0: brain  (FeTS site 1)
  - node 1: prostate (BIDMC)
  - node 2: lung   (site 1)

Uses Flower's simulation API to run everything on one machine.

Usage:
    python -m fedvis.scripts.train_federated --rounds 20
    python -m fedvis.scripts.train_federated --rounds 5 --local_epochs 2
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

import flwr as fl
from flwr.common import ndarrays_to_parameters

from fedvis.models.attention_unet import AttentionUNet3D
from fedvis.federation.client import FedVisClient

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# the three hospital nodes — one per organ
NODES = [
    {'name': 'brain',    'dataset': 'fets',     'site': '1'},
    {'name': 'prostate', 'dataset': 'prostate',  'site': 'BIDMC'},
    {'name': 'lung',     'dataset': 'lung',      'site': '1'},
]


def parse_args():
    p = argparse.ArgumentParser(description='Fed-Vis federated simulation')
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--local_epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--features', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default='outputs/federated')
    p.add_argument('--data_root', type=str, default='data')
    p.add_argument('--cpus', type=int, default=2, help='CPUs per node')
    p.add_argument('--gpus', type=float, default=0.0, help='GPUs per node (fractional ok)')
    return p.parse_args()


def make_client_fn(nodes, args, device):
    """Factory that Flower calls to create each simulated node."""
    from fedvis.scripts.train_local import build_loaders, DATASETS

    def client_fn(cid):
        node = nodes[int(cid)]

        model = AttentionUNet3D(
            in_channels=1, out_channels=1, base_features=args.features
        )

        # reuse the same loader builder from train_local
        class LoaderArgs:
            data = node['dataset']
            site = node['site']
            batch_size = args.batch_size
            seed = args.seed
            data_root = os.path.join(
                args.data_root, DATASETS[node['dataset']]['default_path']
            )

        train_loader, val_loader = build_loaders(LoaderArgs())

        return FedVisClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            name=node['name'],
            cfg={'lr': args.lr, 'local_epochs': args.local_epochs},
            device=device,
        ).to_client()

    return client_fn


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("Fed-Vis Federated Simulation")
    logger.info(f"  rounds={args.rounds}  local_epochs={args.local_epochs}")
    logger.info(f"  nodes: {[n['name'] for n in NODES]}")

    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initial global model weights
    init_model = AttentionUNet3D(in_channels=1, out_channels=1, base_features=args.features)
    init_weights = [v.cpu().numpy() for _, v in init_model.state_dict().items()]

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(NODES),
        min_evaluate_clients=len(NODES),
        min_available_clients=len(NODES),
        initial_parameters=ndarrays_to_parameters(init_weights),
        on_fit_config_fn=lambda rnd: {"local_epochs": args.local_epochs},
    )

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(NODES, args, device),
        num_clients=len(NODES),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": args.cpus, "num_gpus": args.gpus},
    )

    # print results
    logger.info("\n=== Results ===")
    for rnd, metrics in enumerate(history.metrics_distributed.get("dice", [])):
        logger.info(f"  round {rnd}: dice={metrics}")

    if history.losses_distributed:
        logger.info(f"  final loss: {history.losses_distributed[-1]}")

    logger.info("Done.")


if __name__ == '__main__':
    main()

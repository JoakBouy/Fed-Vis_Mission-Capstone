"""Flower server setup and strategy selection."""

import logging
import flwr as fl

logger = logging.getLogger(__name__)


def make_strategy(cfg):
    """Build a Flower aggregation strategy.

    Supports FedAvg out of the box. The weighted averaging
    happens automatically based on each client's sample count.
    """
    name = cfg.get('strategy', 'fedavg')

    if name == 'fedavg':
        return fl.server.strategy.FedAvg(
            fraction_fit=cfg.get('fraction_fit', 1.0),
            fraction_evaluate=cfg.get('fraction_evaluate', 1.0),
            min_fit_clients=cfg.get('min_clients', 3),
            min_evaluate_clients=cfg.get('min_clients', 3),
            min_available_clients=cfg.get('min_clients', 3),
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")

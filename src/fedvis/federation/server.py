"""Flower server setup and strategy selection."""

import logging
import flwr as fl

logger = logging.getLogger(__name__)


def make_strategy(cfg):
    """Build a Flower aggregation strategy.

    Supports 'fedavg' and 'fedprox'.  Note: the proximal term in FedProx is
    applied client-side (see client.py).  At the server level both strategies
    use standard weighted FedAvg aggregation; 'fedprox' simply passes mu to
    clients via the fit config so they can add the proximal regularisation.
    """
    name = cfg.get("strategy", "fedavg")
    num_clients = cfg.get("min_clients", 2)

    common = dict(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    if name in ("fedavg", "fedprox"):
        return fl.server.strategy.FedAvg(**common)

    raise ValueError(f"Unknown strategy: {name!r}. Choose 'fedavg' or 'fedprox'.")

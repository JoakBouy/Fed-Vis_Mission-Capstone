"""Flower client for federated training.

Each client wraps a local model + dataset and exposes
get/set/fit/evaluate to the Flower server.
"""

import logging
from collections import OrderedDict

import numpy as np
import torch

import flwr as fl

from fedvis.models.losses import CombinedLoss, dice_coefficient

logger = logging.getLogger(__name__)


class FedVisClient(fl.client.NumPyClient):
    """One hospital node in the federation."""

    def __init__(self, model, train_loader, val_loader, name, cfg, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.name = name
        self.device = device

        self.criterion = CombinedLoss(dice_weight=1.0, bce_weight=1.0)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.get('lr', 1e-4)
        )

    def get_parameters(self, config):
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, params):
        keys = self.model.state_dict().keys()
        state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)})
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        epochs = config.get('local_epochs', 1)
        self.model.to(self.device)
        self.model.train()

        for ep in range(epochs):
            ep_loss = 0.0
            for vol, mask in self.train_loader:
                vol = vol.to(self.device)
                mask = mask.float().to(self.device)

                out = self.model(vol)
                loss = self.criterion(out, mask)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50.0)
                self.optimizer.step()
                ep_loss += loss.item()

            avg = ep_loss / max(len(self.train_loader), 1)
            logger.info(f"[{self.name}] epoch {ep+1}/{epochs} loss={avg:.4f}")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        dices = []

        with torch.no_grad():
            for vol, mask in self.val_loader:
                vol = vol.to(self.device)
                mask = mask.float().to(self.device)

                out = self.model(vol)
                total_loss += self.criterion(out, mask).item()

                prob = torch.sigmoid(out)
                dices.append(dice_coefficient(prob, mask).item())

        n = max(len(self.val_loader), 1)
        avg_dice = float(np.mean(dices)) if dices else 0.0

        logger.info(f"[{self.name}] eval loss={total_loss/n:.4f} dice={avg_dice:.4f}")
        return total_loss / n, len(self.val_loader.dataset), {"dice": avg_dice}

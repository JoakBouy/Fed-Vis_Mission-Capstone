"""Flower client for federated training.

Each client wraps a local model + dataset and exposes
get/set/fit/evaluate to the Flower server.

Training uses FedProx (proximal regularisation toward the global model),
mixed-precision AMP, and a per-round learning rate supplied by the server.
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
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        # FedProx proximal strength (mu=0 → standard FedAvg)
        self.mu = float(cfg.get("mu", 0.01))

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def get_parameters(self, config):
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, params):
        keys = self.model.state_dict().keys()
        state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)})
        self.model.load_state_dict(state, strict=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        epochs = config.get("local_epochs", 1)
        lr = float(config.get("lr", 3e-4))

        self.model.to(self.device)
        self.model.train()

        # Freeze a snapshot of the global weights for the proximal term
        global_params = [p.clone().detach() for p in self.model.parameters()]

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        criterion = CombinedLoss(dice_weight=1.0, bce_weight=1.0)

        for ep in range(epochs):
            ep_loss = 0.0
            for vol, mask in self.train_loader:
                vol = vol.to(self.device)
                mask = mask.float().to(self.device)

                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = self.model(vol)
                    loss = criterion(out, mask)
                    # FedProx proximal term: (mu/2) * ||w - w_global||^2
                    if self.mu > 0:
                        prox = torch.tensor(0.0, device=self.device)
                        for local_p, global_p in zip(self.model.parameters(), global_params):
                            prox = prox + ((local_p - global_p) ** 2).sum()
                        loss = loss + (self.mu / 2.0) * prox

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                ep_loss += loss.item()

            avg = ep_loss / max(len(self.train_loader), 1)
            logger.info(f"[{self.name}] epoch {ep+1}/{epochs}  loss={avg:.4f}")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        criterion = CombinedLoss(dice_weight=1.0, bce_weight=1.0)
        use_amp = self.device.type == "cuda"
        total_loss = 0.0
        dices, sens_list, prec_list = [], [], []

        with torch.no_grad():
            for vol, mask in self.val_loader:
                vol = vol.to(self.device)
                mask = mask.float().to(self.device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = self.model(vol)
                total_loss += criterion(out, mask).item()

                prob = torch.sigmoid(out)
                pred_bin = (prob > 0.5).float()
                dices.append(dice_coefficient(prob, mask).item())

                tp = (pred_bin * mask).sum().item()
                fn = (mask * (1 - pred_bin)).sum().item()
                fp = (pred_bin * (1 - mask)).sum().item()
                sens_list.append(tp / (tp + fn + 1e-8))
                prec_list.append(tp / (tp + fp + 1e-8))

        n = max(len(self.val_loader), 1)
        avg_dice = float(np.mean(dices)) if dices else 0.0
        avg_sens = float(np.mean(sens_list)) if sens_list else 0.0
        avg_prec = float(np.mean(prec_list)) if prec_list else 0.0

        logger.info(
            f"[{self.name}] eval  loss={total_loss/n:.4f}  "
            f"dice={avg_dice:.4f}  sens={avg_sens:.4f}  prec={avg_prec:.4f}"
        )
        return (
            total_loss / n,
            len(self.val_loader.dataset),
            {"dice": avg_dice, "sensitivity": avg_sens, "precision": avg_prec},
        )

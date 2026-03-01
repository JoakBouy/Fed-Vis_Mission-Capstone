"""Training loop for 3D Attention U-Net segmentation."""

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fedvis.models.losses import CombinedLoss, dice_coefficient

logger = logging.getLogger(__name__)


class Trainer:
    """Runs training and validation for a segmentation model.

    Wraps the standard PyTorch train/val loop with dice tracking,
    checkpoint saving, and TensorBoard logging.
    """

    def __init__(self, model, train_loader, val_loader, cfg, output_dir, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir

        # loss
        self.criterion = CombinedLoss(
            dice_weight=cfg.get('dice_weight', 1.0),
            bce_weight=cfg.get('bce_weight', 1.0),
        )

        # optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.get('lr', 1e-4),
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.get('epochs', 100),
        )

        self.best_dice = 0.0
        self.start_epoch = 0

        # setup output
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    def train(self, num_epochs=None):
        """Run full training."""
        num_epochs = num_epochs or self.cfg.get('epochs', 100)

        logger.info(f"Training for {num_epochs} epochs on {self.device}")
        logger.info(f"  train={len(self.train_loader.dataset)}, "
                     f"val={len(self.val_loader.dataset)}")

        for epoch in range(self.start_epoch, num_epochs):
            t0 = time.time()
            train_loss = self._run_epoch(epoch)
            val_dice = self._evaluate(epoch)
            elapsed = time.time() - t0

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            self.writer.add_scalar('loss/train', train_loss, epoch)
            self.writer.add_scalar('dice/val', val_dice, epoch)
            self.writer.add_scalar('lr', lr, epoch)

            logger.info(
                f"[{epoch+1}/{num_epochs}] "
                f"loss={train_loss:.4f}  dice={val_dice:.4f}  "
                f"lr={lr:.6f}  {elapsed:.0f}s"
            )

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self._save(epoch, tag='best')

            if (epoch + 1) % 10 == 0:
                self._save(epoch, tag=f'ep{epoch}')

        self.writer.close()
        logger.info(f"Done. Best dice={self.best_dice:.4f}")
        return self.best_dice

    def _run_epoch(self, epoch):
        """One training epoch."""
        self.model.train()
        total_loss = 0.0
        n = 0

        for i, (vol, mask) in enumerate(self.train_loader):
            vol = vol.to(self.device)
            mask = mask.float().to(self.device)

            out = self.model(vol)
            loss = self.criterion(out, mask)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=50.0)
            self.optimizer.step()

            total_loss += loss.item()
            n += 1

            if (i + 1) % 20 == 0:
                logger.info(f"  [{epoch+1}] batch {i+1}/{len(self.train_loader)} loss={loss.item():.4f}")

        return total_loss / max(n, 1)

    @torch.no_grad()
    def _evaluate(self, epoch):
        """Compute average dice on validation set."""
        self.model.eval()
        scores = []

        for vol, mask in self.val_loader:
            vol = vol.to(self.device)
            mask = mask.float().to(self.device)

            out = self.model(vol)
            prob = torch.sigmoid(out)
            d = dice_coefficient(prob, mask)
            scores.append(d.item())

        return float(np.mean(scores)) if scores else 0.0

    def _save(self, epoch, tag='checkpoint'):
        """Save model state."""
        path = os.path.join(self.output_dir, 'checkpoints', f'{tag}.pth')
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
        }, path)

    def resume(self, path):
        """Load from a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_dice = ckpt.get('best_dice', 0.0)
        logger.info(f"Resumed from epoch {self.start_epoch}")

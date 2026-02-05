"""Loss functions for medical image segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0, sigmoid: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth, sigmoid=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE loss (50/50 by default).
    Tracks component losses for logging.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        total = dice_weight + bce_weight
        self.dice_weight = dice_weight / total
        self.bce_weight = bce_weight / total

        self.dice_loss = DiceLoss(smooth=smooth, sigmoid=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

        self._last_dice: float = 0.0
        self._last_bce: float = 0.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)

        self._last_dice = dice.item()
        self._last_bce = bce.item()

        return self.dice_weight * dice + self.bce_weight * bce

    def get_component_losses(self) -> dict[str, float]:
        return {"dice": self._last_dice, "bce": self._last_bce}


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    """Compute Dice coefficient (evaluation metric)."""
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

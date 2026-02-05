"""3D Attention U-Net building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """(Conv3D => BN => ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_block = ConvBlock3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv_block(x)
        return x


class UpBlock(nn.Module):
    """Upscaling then double conv with skip connection"""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int | None = None) -> None:
        super().__init__()
        if skip_channels is None:
            skip_channels = out_channels

        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class AttentionGate(nn.Module):
    """
    Attention gate for filtering skip connection features.
    Returns attended features and attention coefficients for visualization.
    """

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int | None = None) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2

        self.W_skip = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta_x = self.W_skip(skip)
        phi_g = self.W_gate(gate)
        phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode="trilinear", align_corners=True)

        f = self.relu(theta_x + phi_g)
        alpha = self.psi(f)
        attended = skip * alpha

        return attended, alpha

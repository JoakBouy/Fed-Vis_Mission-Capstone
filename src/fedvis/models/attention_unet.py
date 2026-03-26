"""3D Attention U-Net for volumetric medical image segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn

from fedvis.models.blocks import AttentionGate, ConvBlock3D, DownBlock, UpBlock


class AttentionUNet3D(nn.Module):
    """
    3D U-Net with attention gates at skip connections.
    
    Based on Oktay et al. (2018) - extended to 3D for volumetric data.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 32,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # Feature progression: 32 -> 64 -> 128 -> 256 -> 512
        features = [base_filters * (2**i) for i in range(5)]

        # Encoder
        self.encoder1 = ConvBlock3D(in_channels, features[0])
        self.encoder2 = DownBlock(features[0], features[1])
        self.encoder3 = DownBlock(features[1], features[2])
        self.encoder4 = DownBlock(features[2], features[3])

        # Bottleneck
        self.bottleneck = DownBlock(features[3], features[4])

        # Attention gates
        self.attention4 = AttentionGate(skip_channels=features[3], gate_channels=features[4])
        self.attention3 = AttentionGate(skip_channels=features[2], gate_channels=features[3])
        self.attention2 = AttentionGate(skip_channels=features[1], gate_channels=features[2])
        self.attention1 = AttentionGate(skip_channels=features[0], gate_channels=features[1])

        # Decoder
        self.decoder4 = UpBlock(features[4], features[3], skip_channels=features[3])
        self.decoder3 = UpBlock(features[3], features[2], skip_channels=features[2])
        self.decoder2 = UpBlock(features[2], features[1], skip_channels=features[1])
        self.decoder1 = UpBlock(features[1], features[0], skip_channels=features[0])

        # Output
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self._attention_maps: list[torch.Tensor] = []

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        self._attention_maps = []

        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        bottleneck = self.dropout(bottleneck)

        # Decoder with attention
        enc4_attn, attn4 = self.attention4(enc4, bottleneck)
        dec4 = self.decoder4(bottleneck, enc4_attn)
        self._attention_maps.append(attn4)

        enc3_attn, attn3 = self.attention3(enc3, dec4)
        dec3 = self.decoder3(dec4, enc3_attn)
        self._attention_maps.append(attn3)

        enc2_attn, attn2 = self.attention2(enc2, dec3)
        dec2 = self.decoder2(dec3, enc2_attn)
        self._attention_maps.append(attn2)

        enc1_attn, attn1 = self.attention1(enc1, dec2)
        dec1 = self.decoder1(dec2, enc1_attn)
        self._attention_maps.append(attn1)

        output = self.final_conv(dec1)

        if return_attention:
            return output, self._attention_maps
        return output

    def get_attention_maps(self) -> list[torch.Tensor]:
        """Get attention maps from last forward pass."""
        return self._attention_maps

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters by component."""
        encoder_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "encoder" in name or "bottleneck" in name
        )
        decoder_params = sum(
            p.numel() for name, p in self.named_parameters() if "decoder" in name
        )
        attention_params = sum(
            p.numel() for name, p in self.named_parameters() if "attention" in name
        )
        total = sum(p.numel() for p in self.parameters())

        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "attention": attention_params,
            "output": total - encoder_params - decoder_params - attention_params,
            "total": total,
        }


def create_attention_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    base_filters: int = 32,
    pretrained: bool = False,
) -> AttentionUNet3D:
    """Factory function for creating Attention U-Net models."""
    if pretrained:
        raise NotImplementedError("Pretrained weights not available yet")

    return AttentionUNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters,
    )

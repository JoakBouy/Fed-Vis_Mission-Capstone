"""Unit Tests for Attention U-Net Model Architecture.

Tests cover:
- Building blocks (ConvBlock3D, DownBlock, UpBlock, AttentionGate)
- Full AttentionUNet3D forward pass
- Attention map generation
- Loss functions
- Shape consistency
"""

import pytest
import torch

from fedvis.models.attention_unet import AttentionUNet3D, create_attention_unet
from fedvis.models.blocks import AttentionGate, ConvBlock3D, DownBlock, UpBlock
from fedvis.models.losses import CombinedLoss, DiceLoss, dice_coefficient


class TestConvBlock3D:
    """Tests for ConvBlock3D building block."""

    def test_shape_preservation(self):
        """ConvBlock3D should preserve spatial dimensions."""
        block = ConvBlock3D(in_channels=1, out_channels=64)
        x = torch.randn(2, 1, 32, 64, 64)
        out = block(x)

        assert out.shape == (2, 64, 32, 64, 64)

    def test_channel_expansion(self):
        """ConvBlock3D should expand channels correctly."""
        block = ConvBlock3D(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 16, 32, 32)
        out = block(x)

        assert out.shape[1] == 128

    def test_gradient_flow(self):
        """Gradients should flow through ConvBlock3D."""
        block = ConvBlock3D(in_channels=1, out_channels=64)
        x = torch.randn(2, 1, 16, 32, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestDownBlock:
    """Tests for DownBlock (encoder block)."""

    def test_spatial_downsampling(self):
        """DownBlock should halve spatial dimensions."""
        block = DownBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 32, 64, 64)
        out = block(x)

        assert out.shape == (2, 128, 16, 32, 32)

    def test_multiple_levels(self):
        """Multiple DownBlocks should progressively downsample."""
        down1 = DownBlock(64, 128)
        down2 = DownBlock(128, 256)

        x = torch.randn(2, 64, 64, 128, 128)
        x = down1(x)
        assert x.shape == (2, 128, 32, 64, 64)

        x = down2(x)
        assert x.shape == (2, 256, 16, 32, 32)


class TestUpBlock:
    """Tests for UpBlock (decoder block)."""

    def test_spatial_upsampling(self):
        """UpBlock should double spatial dimensions."""
        block = UpBlock(in_channels=256, out_channels=128, skip_channels=128)
        x = torch.randn(2, 256, 8, 16, 16)
        skip = torch.randn(2, 128, 16, 32, 32)
        out = block(x, skip)

        assert out.shape == (2, 128, 16, 32, 32)

    def test_skip_concatenation(self):
        """UpBlock should handle skip connection concatenation."""
        block = UpBlock(in_channels=512, out_channels=256, skip_channels=256)
        x = torch.randn(2, 512, 4, 8, 8)
        skip = torch.randn(2, 256, 8, 16, 16)
        out = block(x, skip)

        assert out.shape == (2, 256, 8, 16, 16)


class TestAttentionGate:
    """Tests for AttentionGate module."""

    def test_output_shape(self):
        """AttentionGate should preserve skip connection shape."""
        attn = AttentionGate(skip_channels=64, gate_channels=128)
        skip = torch.randn(2, 64, 32, 64, 64)
        gate = torch.randn(2, 128, 16, 32, 32)

        attended, alpha = attn(skip, gate)

        assert attended.shape == skip.shape
        assert alpha.shape == (2, 1, 32, 64, 64)

    def test_attention_range(self):
        """Attention coefficients should be in [0, 1]."""
        attn = AttentionGate(skip_channels=64, gate_channels=128)
        skip = torch.randn(2, 64, 16, 32, 32)
        gate = torch.randn(2, 128, 8, 16, 16)

        _, alpha = attn(skip, gate)

        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_gradient_flow(self):
        """Gradients should flow through attention gate."""
        attn = AttentionGate(skip_channels=64, gate_channels=128)
        skip = torch.randn(2, 64, 16, 32, 32, requires_grad=True)
        gate = torch.randn(2, 128, 8, 16, 16, requires_grad=True)

        attended, _ = attn(skip, gate)
        loss = attended.sum()
        loss.backward()

        assert skip.grad is not None
        assert gate.grad is not None


class TestAttentionUNet3D:
    """Tests for full AttentionUNet3D architecture."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return AttentionUNet3D(in_channels=1, out_channels=1, base_filters=32)

    def test_forward_pass(self, model):
        """Model should produce output of same spatial size as input."""
        x = torch.randn(2, 1, 32, 64, 64)
        out = model(x)

        assert out.shape == (2, 1, 32, 64, 64)

    def test_attention_maps_return(self, model):
        """Model should return attention maps when requested."""
        x = torch.randn(2, 1, 32, 64, 64)
        out, attn_maps = model(x, return_attention=True)

        assert len(attn_maps) == 4
        assert out.shape == (2, 1, 32, 64, 64)

    def test_get_attention_maps(self, model):
        """Attention maps should be accessible after forward pass."""
        x = torch.randn(2, 1, 32, 64, 64)
        _ = model(x)
        attn_maps = model.get_attention_maps()

        assert len(attn_maps) == 4
        for attn in attn_maps:
            assert attn.min() >= 0.0
            assert attn.max() <= 1.0

    def test_count_parameters(self, model):
        """Parameter counting should work correctly."""
        params = model.count_parameters()

        assert "encoder" in params
        assert "decoder" in params
        assert "attention" in params
        assert "total" in params
        assert params["total"] > 0

    def test_different_input_sizes(self, model):
        """Model should handle different input sizes (powers of 2)."""
        for size in [(32, 64, 64), (64, 64, 64), (16, 128, 128)]:
            x = torch.randn(1, 1, *size)
            out = model(x)
            assert out.shape == (1, 1, *size)

    def test_batch_independence(self, model):
        """Different batch items should produce independent outputs."""
        x1 = torch.randn(1, 1, 32, 64, 64)
        x2 = torch.randn(1, 1, 32, 64, 64)
        x_batch = torch.cat([x1, x2], dim=0)

        model.eval()
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            out_batch = model(x_batch)

        assert torch.allclose(out1, out_batch[0:1], atol=1e-5)
        assert torch.allclose(out2, out_batch[1:2], atol=1e-5)


class TestCreateAttentionUNet:
    """Tests for model factory function."""

    def test_default_creation(self):
        """Factory should create model with default settings."""
        model = create_attention_unet()

        assert isinstance(model, AttentionUNet3D)

    def test_custom_channels(self):
        """Factory should respect custom channel settings."""
        model = create_attention_unet(in_channels=4, out_channels=3)
        x = torch.randn(1, 4, 32, 64, 64)
        out = model(x)

        assert out.shape == (1, 3, 32, 64, 64)

    def test_pretrained_not_implemented(self):
        """Pretrained weights should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            create_attention_unet(pretrained=True)


class TestDiceLoss:
    """Tests for Dice loss function."""

    def test_perfect_prediction(self):
        """Dice loss should be 0 for perfect prediction."""
        loss_fn = DiceLoss()
        # Perfect match
        pred = torch.ones(2, 1, 16, 32, 32) * 10  # High logit
        target = torch.ones(2, 1, 16, 32, 32)

        loss = loss_fn(pred, target)
        assert loss.item() < 0.01  # Near zero

    def test_worst_prediction(self):
        """Dice loss should be high for complete mismatch."""
        loss_fn = DiceLoss()
        # Complete mismatch
        pred = torch.ones(2, 1, 16, 32, 32) * -10  # Low logit (predicts 0)
        target = torch.ones(2, 1, 16, 32, 32)

        loss = loss_fn(pred, target)
        assert loss.item() > 0.9  # Near 1

    def test_gradient_flow(self):
        """Gradients should flow through Dice loss."""
        loss_fn = DiceLoss()
        pred = torch.randn(2, 1, 16, 32, 32, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 16, 32, 32)).float()

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None


class TestCombinedLoss:
    """Tests for combined Dice + BCE loss."""

    def test_combined_loss_range(self):
        """Combined loss should produce reasonable values."""
        loss_fn = CombinedLoss()
        pred = torch.randn(2, 1, 16, 32, 32)
        target = torch.randint(0, 2, (2, 1, 16, 32, 32)).float()

        loss = loss_fn(pred, target)

        assert loss.item() >= 0
        assert loss.isfinite()

    def test_component_tracking(self):
        """Should track individual loss components."""
        loss_fn = CombinedLoss()
        pred = torch.randn(2, 1, 16, 32, 32)
        target = torch.randint(0, 2, (2, 1, 16, 32, 32)).float()

        _ = loss_fn(pred, target)
        components = loss_fn.get_component_losses()

        assert "dice" in components
        assert "bce" in components

    def test_weight_normalization(self):
        """Weights should be normalized to sum to 1."""
        loss_fn = CombinedLoss(dice_weight=2.0, bce_weight=2.0)

        assert abs(loss_fn.dice_weight + loss_fn.bce_weight - 1.0) < 1e-6


class TestDiceCoefficient:
    """Tests for Dice coefficient metric."""

    def test_perfect_overlap(self):
        """Dice should be 1 for perfect overlap."""
        pred = torch.ones(2, 1, 16, 32, 32)
        target = torch.ones(2, 1, 16, 32, 32)

        dice = dice_coefficient(pred, target)
        assert dice.item() > 0.99

    def test_no_overlap(self):
        """Dice should be near 0 for no overlap."""
        pred = torch.zeros(2, 1, 16, 32, 32)
        target = torch.ones(2, 1, 16, 32, 32)

        dice = dice_coefficient(pred, target)
        assert dice.item() < 0.01

    def test_partial_overlap(self):
        """Dice should be between 0 and 1 for partial overlap."""
        pred = torch.zeros(2, 1, 16, 32, 32)
        pred[:, :, :8] = 1.0  # Half prediction
        target = torch.ones(2, 1, 16, 32, 32)

        dice = dice_coefficient(pred, target)
        assert 0.3 < dice.item() < 0.7


class TestGPUCompatibility:
    """Tests for GPU compatibility (if available)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_gpu(self):
        """Model should work on GPU."""
        model = AttentionUNet3D(in_channels=1, out_channels=1, base_features=32)
        model = model.cuda()
        x = torch.randn(2, 1, 32, 64, 64).cuda()

        out = model(x)
        assert out.device.type == "cuda"
        assert out.shape == (2, 1, 32, 64, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss_on_gpu(self):
        """Loss should work on GPU."""
        loss_fn = CombinedLoss().cuda()
        pred = torch.randn(2, 1, 16, 32, 32).cuda()
        target = torch.randint(0, 2, (2, 1, 16, 32, 32)).float().cuda()

        loss = loss_fn(pred, target)
        assert loss.device.type == "cuda"

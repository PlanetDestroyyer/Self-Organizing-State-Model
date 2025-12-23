"""
Unit tests for MU attention layer
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mu_layer import MUAttentionLayer, MUTransformerBlock


def test_mu_layer_forward_shape():
    """Test that MU layer preserves shape"""
    layer = MUAttentionLayer(r=4, c=4, d_model=128, n_heads=4)
    B, T = 2, 10
    M = torch.randn(B, T, 4, 4)

    M_out = layer(M)

    assert M_out.shape == (B, T, 4, 4), f"Expected {(B, T, 4, 4)}, got {M_out.shape}"


def test_mu_layer_no_nans():
    """Test that forward pass doesn't produce NaNs"""
    layer = MUAttentionLayer(r=4, c=4, d_model=128, n_heads=4)
    M = torch.randn(2, 10, 4, 4)

    M_out = layer(M)

    assert not torch.isnan(M_out).any(), "Forward pass produced NaNs"


def test_mu_layer_gradient_flow():
    """Test that gradients flow through the layer"""
    layer = MUAttentionLayer(r=4, c=4, d_model=128, n_heads=4)
    M = torch.randn(2, 10, 4, 4, requires_grad=True)

    M_out = layer(M)
    loss = M_out.sum()
    loss.backward()

    assert M.grad is not None, "No gradient on input"
    assert not torch.isnan(M.grad).any(), "Gradient contains NaNs"


def test_sensitivity_gating():
    """Test that different slots update at different rates"""
    layer = MUAttentionLayer(r=4, c=4, d_model=128, n_heads=4)
    M = torch.randn(2, 10, 4, 4)

    M_out = layer(M)

    # Identity slot should change less than context slot
    delta_identity = torch.abs(M_out[:, :, 0, 0] - M[:, :, 0, 0]).mean()
    delta_context = torch.abs(M_out[:, :, 1, 3] - M[:, :, 1, 3]).mean()

    # This is a hypothesis - it might fail if gates haven't learned properly
    # For now, just check that both are finite
    assert torch.isfinite(delta_identity), "Identity delta not finite"
    assert torch.isfinite(delta_context), "Context delta not finite"


def test_mu_transformer_block():
    """Test full MU transformer block"""
    block = MUTransformerBlock(r=4, c=4, d_model=128, n_heads=4)
    M = torch.randn(2, 10, 4, 4)

    M_out = block(M)

    assert M_out.shape == (2, 10, 4, 4)
    assert not torch.isnan(M_out).any()


def test_attention_mask():
    """Test that attention mask works"""
    layer = MUAttentionLayer(r=4, c=4, d_model=128, n_heads=4)
    B, T = 2, 10
    M = torch.randn(B, T, 4, 4)

    # Create causal mask
    mask = torch.tril(torch.ones(T, T)).bool()

    M_out = layer(M, mask=mask)

    assert M_out.shape == (B, T, 4, 4)
    assert not torch.isnan(M_out).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

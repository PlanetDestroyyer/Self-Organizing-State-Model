"""
Gradient flow tests
"""
import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mu_transformer import MUTransformer
from src.models.config import MUConfig


def test_gradient_flow_through_model():
    """Test that gradients flow through entire model"""
    config = MUConfig(vocab_size=1000, r=4, c=4, n_layers=2, max_seq_len=128)
    model = MUTransformer(config)

    input_ids = torch.randint(0, 1000, (2, 10))
    targets = torch.randint(0, 1000, (2, 10))

    logits, _ = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))

    loss.backward()

    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"


def test_gradient_magnitudes():
    """Test that gradient magnitudes are reasonable"""
    config = MUConfig(vocab_size=1000, r=4, c=4, n_layers=2, max_seq_len=128)
    model = MUTransformer(config)

    input_ids = torch.randint(0, 1000, (2, 10))
    targets = torch.randint(0, 1000, (2, 10))

    logits, _ = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))

    loss.backward()

    # Collect gradient norms
    grad_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

            # Gradients shouldn't be too large
            assert grad_norm < 1000, f"Very large gradient for {name}: {grad_norm}"

    # At least some gradients should be non-zero
    assert any(g > 1e-8 for g in grad_norms), "All gradients are effectively zero"


def test_backward_forward_consistency():
    """Test that multiple forward-backward passes are consistent"""
    config = MUConfig(vocab_size=1000, r=4, c=4, n_layers=2, max_seq_len=128)
    model = MUTransformer(config)

    input_ids = torch.randint(0, 1000, (2, 10))
    targets = torch.randint(0, 1000, (2, 10))

    # First pass
    logits1, _ = model(input_ids)
    loss1 = F.cross_entropy(logits1.view(-1, 1000), targets.view(-1))

    # Second pass (should be identical)
    logits2, _ = model(input_ids)
    loss2 = F.cross_entropy(logits2.view(-1, 1000), targets.view(-1))

    assert torch.allclose(logits1, logits2), "Forward pass not deterministic"
    assert torch.allclose(loss1, loss2), "Loss not deterministic"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

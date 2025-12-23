"""
Training tests
"""
import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mu_transformer import MUTransformer
from src.models.baseline_transformer import BaselineTransformer
from src.models.config import MUConfig, BaselineConfig


def test_training_step():
    """Test that one training step executes without errors"""
    config = MUConfig(vocab_size=1000, r=4, c=4, n_layers=2, max_seq_len=128)
    model = MUTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, 1000, (4, 10))
    targets = torch.randint(0, 1000, (4, 10))

    logits, _ = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"


def test_overfitting_single_batch():
    """Test that model can overfit a single batch (sanity check)"""
    config = MUConfig(vocab_size=100, r=4, c=4, n_layers=2, d_model=64, max_seq_len=128)
    model = MUTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    input_ids = torch.randint(0, 100, (2, 10))
    targets = torch.randint(0, 100, (2, 10))

    initial_loss = None
    for step in range(100):
        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, 100), targets.view(-1))

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5, \
        f"Model didn't overfit: {initial_loss:.4f} -> {final_loss:.4f}"


def test_baseline_training_step():
    """Test baseline model training step"""
    config = BaselineConfig(vocab_size=1000, d_model=128, n_layers=2, max_seq_len=128)
    model = BaselineTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, 1000, (4, 10))
    targets = torch.randint(0, 1000, (4, 10))

    logits, _ = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_gradient_clipping():
    """Test gradient clipping"""
    config = MUConfig(vocab_size=1000, r=4, c=4, n_layers=2, max_seq_len=128)
    model = MUTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, 1000, (4, 10))
    targets = torch.randint(0, 1000, (4, 10))

    logits, _ = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))

    loss.backward()

    # Clip gradients
    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # Total norm should be <= max_norm after clipping
    assert total_norm <= max_norm or torch.isclose(total_norm, torch.tensor(max_norm), atol=1e-4)

    optimizer.step()
    optimizer.zero_grad()


def test_loss_functions():
    """Test custom loss functions"""
    from src.training.losses import MUTransformerLoss, LanguageModelingLoss

    # LM loss
    lm_loss = LanguageModelingLoss(ignore_index=-100)
    logits = torch.randn(2, 10, 1000)
    labels = torch.randint(0, 1000, (2, 10))

    loss = lm_loss(logits, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0

    # MU loss
    mu_loss = MUTransformerLoss(lambda_lm=1.0, lambda_inv=0.5)
    MU = torch.randn(2, 10, 4, 4)

    total_loss, loss_dict = mu_loss(logits, labels, MU)
    assert torch.isfinite(total_loss)
    assert 'total' in loss_dict
    assert 'lm' in loss_dict
    assert 'invariance' in loss_dict


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

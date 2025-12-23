"""
Shape consistency tests
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mu_transformer import MUTransformer, MUTransformerLM
from src.models.baseline_transformer import BaselineTransformer, BaselineTransformerLM
from src.models.config import MUConfig, BaselineConfig


def test_token_to_mu_shape():
    """Test token embedding to MU conversion"""
    config = MUConfig(vocab_size=1000, r=4, c=4, max_seq_len=128)
    model = MUTransformer(config)

    input_ids = torch.randint(0, 1000, (2, 10))
    logits, MU = model(input_ids)

    assert logits.shape == (2, 10, 1000), f"Expected logits shape (2, 10, 1000), got {logits.shape}"
    assert MU.shape == (2, 10, 4, 4), f"Expected MU shape (2, 10, 4, 4), got {MU.shape}"


def test_batch_independence():
    """Test that batch elements are processed independently"""
    from src.models.mu_layer import MUAttentionLayer

    layer = MUAttentionLayer(r=4, c=4, d_model=128, n_heads=4)

    M_single = torch.randn(1, 10, 4, 4)
    M_batch = M_single.repeat(4, 1, 1, 1)

    out_single = layer(M_single)
    out_batch = layer(M_batch)

    # All batch elements should be identical
    for i in range(4):
        assert torch.allclose(out_batch[i], out_single[0], atol=1e-5), \
            f"Batch element {i} differs from single"


def test_baseline_transformer_shape():
    """Test baseline transformer shapes"""
    config = BaselineConfig(vocab_size=1000, d_model=128, max_seq_len=128)
    model = BaselineTransformer(config)

    input_ids = torch.randint(0, 1000, (2, 10))
    logits, _ = model(input_ids)

    assert logits.shape == (2, 10, 1000)


def test_causal_masking():
    """Test causal masking in language models"""
    config = MUConfig(vocab_size=1000, r=4, c=4, max_seq_len=128)
    model = MUTransformerLM(config)

    input_ids = torch.randint(0, 1000, (2, 10))
    logits, MU = model(input_ids)

    assert logits.shape == (2, 10, 1000)
    assert MU.shape == (2, 10, 4, 4)


def test_different_sequence_lengths():
    """Test model with different sequence lengths"""
    config = MUConfig(vocab_size=1000, r=4, c=4, max_seq_len=128)
    model = MUTransformer(config)

    for seq_len in [5, 10, 20, 50]:
        input_ids = torch.randint(0, 1000, (2, seq_len))
        logits, MU = model(input_ids)

        assert logits.shape == (2, seq_len, 1000)
        assert MU.shape == (2, seq_len, 4, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

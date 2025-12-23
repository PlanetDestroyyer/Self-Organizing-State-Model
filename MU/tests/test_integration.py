"""
Integration tests
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mu_transformer import MUTransformer
from src.models.config import MUConfig
from src.training.trainer import Trainer
from torch.utils.data import TensorDataset, DataLoader


def test_minimal_training_loop():
    """Test a minimal training loop"""
    config = MUConfig(vocab_size=100, r=4, c=4, n_layers=2, d_model=64, max_seq_len=32)
    model = MUTransformer(config)

    # Create tiny dataset
    input_ids = torch.randint(0, 100, (20, 16))
    labels = torch.randint(0, 100, (20, 16))

    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=4)

    # Create trainer
    train_config = {
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'lambda_lm': 1.0,
        'lambda_inv': 0.1,
        'log_interval': 10,
        'eval_interval': 100,
        'save_interval': 100,
        'mixed_precision': False,
        'checkpoint_dir': 'results/test_checkpoints'
    }

    trainer = Trainer(model, train_config, device='cpu')

    # Manually train for a few steps
    model.train()
    for batch in dataloader:
        batch_dict = {
            'input_ids': batch[0],
            'labels': batch[1]
        }
        metrics = trainer.train_step(batch_dict)

        assert 'total' in metrics
        assert 'perplexity' in metrics
        assert torch.isfinite(torch.tensor(metrics['total']))
        break  # Just test one step


def test_model_save_load():
    """Test model saving and loading"""
    from src.utils.checkpoint import save_model_only, load_model_only
    import tempfile
    import os

    config = MUConfig(vocab_size=100, r=4, c=4, n_layers=2, max_seq_len=32)
    model1 = MUTransformer(config)

    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'model.pt')
        save_model_only(model1, save_path, config={'test': True})

        # Load model
        model2 = MUTransformer(config)
        loaded_config = load_model_only(model2, save_path, device='cpu')

        # Check that weights match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Loaded weights don't match"

        assert loaded_config['test'] == True


def test_full_pipeline():
    """Test full training pipeline with tiny data"""
    config = MUConfig(vocab_size=50, r=4, c=4, n_layers=1, d_model=32, max_seq_len=16)
    model = MUTransformer(config)

    # Create tiny dataset
    input_ids = torch.randint(0, 50, (10, 8))
    labels = torch.randint(0, 50, (10, 8))

    # Create dataloader that returns proper format
    class SimpleDataset:
        def __init__(self, input_ids, labels):
            self.input_ids = input_ids
            self.labels = labels

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'labels': self.labels[idx]
            }

    dataset = SimpleDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=2)

    # Train for 1 epoch
    train_config = {
        'learning_rate': 1e-3,
        'lambda_lm': 1.0,
        'lambda_inv': 0.0,
        'log_interval': 1,
        'eval_interval': 100,
        'save_interval': 100,
        'mixed_precision': False,
        'checkpoint_dir': 'results/test_checkpoints',
        'num_epochs': 1
    }

    trainer = Trainer(model, train_config, device='cpu')
    trainer.train(dataloader, val_loader=None, num_epochs=1)

    # Model should have trained
    assert trainer.global_step > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

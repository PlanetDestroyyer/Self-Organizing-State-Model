#!/usr/bin/env python3
"""
Training script for MU Transformer

Usage:
    python scripts/train.py --config configs/mu_small.yaml --model mu --seed 42
    python scripts/train.py --config configs/baseline_small.yaml --model baseline --seed 42
"""
import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mu_transformer import MUTransformerLM
from src.models.baseline_transformer import BaselineTransformerLM
from src.models.config import MUConfig, BaselineConfig, get_config_from_dict
from src.training.trainer import Trainer
from src.data.datasets import get_dataloaders
from src.utils.seed import set_seed
from src.utils.logging_utils import setup_logger, log_model_info


def main():
    parser = argparse.ArgumentParser(description='Train MU Transformer or Baseline')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, choices=['mu', 'baseline'], required=True,
                       help='Model type to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with minimal data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(
        name='train',
        log_file=f'results/logs/{args.model}_training.log'
    )

    logger.info("=" * 60)
    logger.info("MU TRANSFORMER TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model type: {args.model}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Test mode: {args.test_mode}")

    # Set seed
    set_seed(args.seed)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {args.config}")

    # Create model
    model_config = get_config_from_dict(config['model'], model_type=args.model)

    if args.model == 'mu':
        model = MUTransformerLM(model_config)
        logger.info("Created MU Transformer model")
    else:
        model = BaselineTransformerLM(model_config)
        logger.info("Created Baseline Transformer model")

    # Log model info
    log_model_info(logger, model)

    # Load data
    logger.info("Loading data...")
    data_config = config.get('data', {})
    data_config['batch_size'] = config['training'].get('batch_size', 32)
    data_config['sequence_length'] = config['model'].get('max_seq_len', 256)

    train_loader, val_loader, test_loader = get_dataloaders(data_config, test_mode=args.test_mode)

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Setup training config
    training_config = {
        **config['training'],
        'checkpoint_dir': f'results/checkpoints/{args.model}',
        'log_file': f'results/logs/{args.model}_training.log'
    }

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(model, training_config, device=args.device)

    # Train
    num_epochs = config['training'].get('num_epochs', 10)
    if args.test_mode:
        num_epochs = 1

    logger.info(f"Starting training for {num_epochs} epochs...")
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    # Final evaluation
    logger.info("Running final evaluation...")
    test_metrics = trainer.evaluate(test_loader)

    logger.info("=" * 60)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Perplexity: {test_metrics['perplexity']:.2f}")

    # Save final model
    from src.utils.checkpoint import save_model_only
    save_model_only(
        model,
        f'results/checkpoints/{args.model}_final.pt',
        config=config
    )

    logger.info("Training complete!")


if __name__ == '__main__':
    main()

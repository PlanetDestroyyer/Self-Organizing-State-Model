"""
MU-SOTA Transformer - Training Entry Point

Clean wrapper that uses the modular implementation from src/

Usage:
    python run_colab.py
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizers warning

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import logging

# Import from modular structure
from src.config import MUSOTAConfig
from src.models import MUSOTATransformer
from src.data import WikiTextBPEDataset
from src.training import train_epoch, evaluate, generate_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""

    # Initialize configuration
    config = MUSOTAConfig()

    logger.info("=" * 80)
    logger.info("MU-SOTA TRANSFORMER - Production Training")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  • Matrix: 8×8 (16 semantic blocks)")
    logger.info(f"  • Layers: {config.n_layers}")
    logger.info(f"  • Vocab: {config.vocab_size}")
    logger.info(f"  • Mixed Precision: {config.use_mixed_precision}")
    logger.info(f"  • Device: {config.device}")
    logger.info("=" * 80)

    # Load data
    try:
        train_dataset = WikiTextBPEDataset('train', config.max_seq_len, vocab_size=config.vocab_size)
        val_dataset = WikiTextBPEDataset('validation', config.max_seq_len, tokenizer=train_dataset.tokenizer)

        # Update config with actual vocab size
        config.vocab_size = train_dataset.vocab_size

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=2)

        logger.info(f"Dataset loaded:")
        logger.info(f"  • Train: {len(train_dataset):,} sequences")
        logger.info(f"  • Val: {len(val_dataset):,} sequences")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Create model
    model = MUSOTATransformer(config).to(config.device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=total_steps,
        pct_start=config.warmup_steps/total_steps
    )
    # Create GradScaler for mixed precision
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    scaler = GradScaler(device_type, enabled=config.use_mixed_precision)

    # Training loop
    best_perplexity = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler, config.device, epoch, config.num_epochs)
        val_metrics = evaluate(model, val_loader, config.device)

        logger.info(f"\nEpoch {epoch}:")
        logger.info(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']*100:.2f}%")
        logger.info(f"  Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']*100:.2f}%, PPL={val_metrics['perplexity']:.2f}")

        # Generate sample text AFTER EACH EPOCH to verify it's working
        logger.info(f"\n{'='*80}")
        logger.info(f"TEXT GENERATION TEST - EPOCH {epoch}")
        logger.info(f"{'='*80}")
        test_prompts = ["The quick brown", "Once upon a time", "In the beginning"]
        for prompt in test_prompts:
            try:
                generated = generate_text(model, train_dataset, prompt, max_length=30, device=config.device)
                logger.info(f"  Prompt: '{prompt}'")
                logger.info(f"  Generated: '{generated}'")
                logger.info("-" * 80)
            except Exception as e:
                logger.error(f"  Error generating from '{prompt}': {e}")
        logger.info(f"{'='*80}\n")

        # Save best model
        if val_metrics['perplexity'] < best_perplexity:
            best_perplexity = val_metrics['perplexity']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'perplexity': best_perplexity,
            }, 'mu_sota_best.pt')
            train_dataset.tokenizer.save('mu_sota_tokenizer.json')
            logger.info(f"  ✓ Saved best model (PPL={best_perplexity:.2f})")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

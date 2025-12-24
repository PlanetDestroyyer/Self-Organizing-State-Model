#!/usr/bin/env python3
"""
Fixed Training Script for SOSM
Addresses generation quality issues with gradual complexity scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import argparse

from state_core import StateCorePipeline
from transformers import GPT2Tokenizer
from datasets import load_dataset


# ============================================================================
# CONFIGURATION (FIXED)
# ============================================================================

def get_fixed_config(stage=0):
    """
    Get properly sized config for each stage.

    Stage 0: MU only (simple baseline)
    Stage 1: MU + TEMPORAL
    Stage 2: MU + TEMPORAL + K-1
    Stage 3: Full system
    """
    config = {
        'stage': stage,
        'components': {
            'mu': {
                'vocab_size': 50257,
                'embed_dim': 512,  # 64 → 512 (8x increase)
                'use_full_model': False,  # Start simple!
                'mu_layers': 1,
                'max_seq_len': 256,
            },
            'temporal': {
                'time_dim': 128,  # 32 → 128 (4x increase)
                'learning_mode': 'gradient',
            },
            'k1': {
                'use_hierarchical_tree': False,  # Simplified K-1
            },
            'graph': {
                'sequential_edges': True,
                'semantic_edges': False,  # Disable for now
                'random_shortcuts': 0.0,
                'semantic_threshold': 0.3,
            }
        },
        'model': {
            'hidden_dim': 1024,  # 256 → 1024 (4x increase)
            'n_layers': 6,  # Keep at 6 for faster training
            'n_heads': 8,  # 4 → 8
            'dropout': 0.1
        }
    }

    return config


# ============================================================================
# DATA LOADING
# ============================================================================

class WikiTextDataset:
    """WikiText-2 dataset with proper tokenization"""

    def __init__(self, split='train', seq_len=256, tokenizer=None):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        print(f"Loading WikiText-2 {split}...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Tokenize all text
        all_tokens = []
        for item in tqdm(dataset, desc="Tokenizing"):
            if len(item['text'].strip()) > 0:
                tokens = tokenizer.encode(item['text'], add_special_tokens=False)
                all_tokens.extend(tokens)

        # Create sequences
        self.sequences = []
        for i in range(0, len(all_tokens) - seq_len - 1, seq_len // 2):
            seq = all_tokens[i:i + seq_len + 1]
            if len(seq) == seq_len + 1:
                self.sequences.append(torch.tensor(seq, dtype=torch.long))

        print(f"Created {len(self.sequences)} sequences of length {seq_len}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'input_ids': seq[:-1],
            'labels': seq[1:]
        }


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, stage,
                max_steps=None, log_every=100):
    """Train for one epoch with proper monitoring"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Training Stage {stage}')

    for i, batch in enumerate(pbar):
        if max_steps and i >= max_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward
        logits, state = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            labels.reshape(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check for gradient issues
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item()

        if math.isnan(total_grad_norm) or math.isinf(total_grad_norm):
            print(f"\n⚠️  WARNING: Gradient is {total_grad_norm}, skipping step")
            continue

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # K-1 attribution (if Stage 2+)
        if stage >= 2:
            model.backward_with_k1(loss, state, i)

        optimizer.step()
        scheduler.step()

        # Logging
        total_loss += loss.item()
        num_batches += 1

        if i % log_every == 0:
            avg_loss = total_loss / max(1, num_batches)
            ppl = math.exp(min(avg_loss, 20))

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'ppl': f'{ppl:.1f}',
                'grad': f'{total_grad_norm:.2f}'
            })

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(model, dataloader, device, stage):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f'Evaluating Stage {stage}'):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            labels.reshape(-1)
        )

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    perplexity = math.exp(min(avg_loss, 20))

    return {'loss': avg_loss, 'perplexity': perplexity}


def test_generation(model, tokenizer, device, stage):
    """Test generation quality"""
    model.eval()

    prompts = [
        "The capital of India is",
        "Once upon a time",
        "The quick brown fox"
    ]

    print("\n" + "="*70)
    print(f"GENERATION TEST - Stage {stage}")
    print("="*70)

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Greedy decoding
        with torch.no_grad():
            for _ in range(20):
                logits, _ = model(input_ids)
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"\nPrompt: '{prompt}'")
        print(f"Output: {output}")

    print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0, help='Training stage (0-3)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--max-steps', type=int, default=None, help='Max steps per epoch')
    parser.add_argument('--save-path', type=str, default='sosm_checkpoint.pt', help='Save path')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("\nLoading datasets...")
    train_dataset = WikiTextDataset('train', seq_len=256, tokenizer=tokenizer)
    val_dataset = WikiTextDataset('validation', seq_len=256, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=2
    )

    # Create model
    print(f"\nInitializing SOSM (Stage {args.stage})...")
    config = get_fixed_config(args.stage)
    model = StateCorePipeline(config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(2000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print("\n" + "="*70)
    print(f"TRAINING STAGE {args.stage}")
    print("="*70)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.stage, args.max_steps
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, args.stage)

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
            }, args.save_path)
            print(f"  ✓ Saved checkpoint (val_loss={val_metrics['loss']:.4f})")

        # Test generation
        if (epoch + 1) % 2 == 0:
            test_generation(model, tokenizer, device, args.stage)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best perplexity: {math.exp(best_val_loss):.2f}")
    print(f"Model saved to: {args.save_path}")


if __name__ == '__main__':
    main()

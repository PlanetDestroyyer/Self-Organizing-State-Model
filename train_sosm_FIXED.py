#!/usr/bin/env python3
"""
FIXED SOSM Training Script

Uses best practices from successful MU and TEMPORAL experiments:
- AdamW optimizer (proven for transformers)
- Mixed precision training (FP16)
- Gradient clipping (prevents explosion)
- OneCycleLR or Cosine schedule with warmup
- Proper logging
- Incremental staging (0 → 1 → 2 → 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
from tqdm import tqdm
import argparse
from pathlib import Path

from state_core import StateCorePipeline
from transformers import GPT2Tokenizer
from datasets import load_dataset


# ============================================================================
# DATASET
# ============================================================================

class WikiTextDataset:
    """WikiText dataset with GPT-2 tokenization"""

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

        print(f"Created {len(self.sequences)} sequences")

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

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, stage, epoch, total_epochs):
    """Train for one epoch with proper monitoring"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Stage {stage}]')

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Mixed precision forward pass
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            logits, state = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                labels.reshape(-1)
            )

        # Backward with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping (CRITICAL for stability!)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Check for NaN/Inf gradients
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"\n⚠️  WARNING: Bad gradients detected (grad_norm={grad_norm}), skipping step")
            continue

        # K-1 attribution (if Stage 2+)
        if stage >= 2:
            model.backward_with_k1(loss, state, batch_idx)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Logging
        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            avg_loss = total_loss / max(1, num_batches)
            ppl = math.exp(min(avg_loss, 20))
            lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'ppl': f'{ppl:.1f}',
                'lr': f'{lr:.2e}',
                'grad': f'{grad_norm:.2f}'
            })

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc='Evaluating'):
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


@torch.no_grad()
def test_generation(model, tokenizer, device, stage):
    """Quick generation test"""
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

        # Greedy decoding (simple test)
        for _ in range(15):
            logits, _ = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"\n'{prompt}' →\n  {output}")

    print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0, help='Training stage (0-3)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=256, help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"SOSM TRAINING - Stage {args.stage}")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("\nLoading datasets...")
    train_dataset = WikiTextDataset('train', seq_len=args.seq_len, tokenizer=tokenizer)
    val_dataset = WikiTextDataset('validation', seq_len=args.seq_len, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True
    )

    # Create model with FIXED defaults
    print(f"\nInitializing SOSM (Stage {args.stage})...")
    config = {
        'stage': args.stage,
        'components': {
            'mu': {
                'vocab_size': 50257,
                'embed_dim': 512,  # INCREASED from 64
                'use_full_model': False,  # Start simple
            },
            'temporal': {
                'time_dim': 256,  # INCREASED from 32
            },
            'k1': {},
            'graph': {
                'sequential_edges': True,
                'semantic_edges': False,  # Disabled
                'random_shortcuts': 0.0,  # Disabled
            }
        },
        'model': {
            'hidden_dim': 1024,  # INCREASED from 256
            'n_layers': 6,
            'n_heads': 8,  # INCREASED from 4
            'dropout': 0.1
        }
    }

    model = StateCorePipeline(config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # Optimizer (AdamW like MU and TEMPORAL)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),  # Standard for transformers
        eps=1e-8,
        weight_decay=0.01
    )

    # Scheduler (OneCycleLR like MU)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )

    # Mixed precision
    scaler = GradScaler('cuda')

    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    best_val_loss = float('inf')
    save_path = f'sosm_stage{args.stage}_FIXED.pt'

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, args.stage, epoch, args.epochs
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
                'vocab_size': 50257
            }, save_path)
            print(f"  ✓ Saved checkpoint (val_loss={val_metrics['loss']:.4f})")

        # Test generation every 2 epochs
        if epoch % 2 == 0:
            test_generation(model, tokenizer, device, args.stage)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best perplexity: {math.exp(best_val_loss):.2f}")
    print(f"Model saved to: {save_path}")


if __name__ == '__main__':
    main()

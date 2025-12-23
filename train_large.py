#!/usr/bin/env python3
"""
SOSM Large-Scale Training Script

Trains SOSM and Baseline with:
- Larger model dimensions (256 vs 96)
- More MU layers (4 vs 2)
- 50 epochs with checkpointing
- Built-in generation test

Usage:
    python train_large.py --epochs 50 --model-dim 256 --mu-layers 4
    python train_large.py --epochs 10 --quick  # Quick test
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from sosm_data import MultiDomainDataset
from state_core import StateCorePipeline
from test_base import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(description='SOSM Large-Scale Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=64, help='Sequence length')
    parser.add_argument('--model-dim', type=int, default=256, help='Model hidden dimension')
    parser.add_argument('--embed-dim', type=int, default=128, help='MU embedding dimension (must be square)')
    parser.add_argument('--time-dim', type=int, default=64, help='TEMPORAL dimension')
    parser.add_argument('--mu-layers', type=int, default=4, help='Number of MU block-attention layers')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--checkpoint-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (smaller dataset)')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline training')
    return parser.parse_args()


def create_sosm(args, vocab_size):
    """Create SOSM with larger configuration."""
    config = {
        'stage': 3,
        'components': {
            'mu': {
                'vocab_size': vocab_size,
                'embed_dim': args.embed_dim,  # Larger: 128 vs 64
                'max_seq_len': args.seq_length,
                'use_full_model': True,
                'mu_layers': args.mu_layers,  # More: 4 vs 2
            },
            'temporal': {
                'time_dim': args.time_dim,  # Larger: 64 vs 32
                'learning_mode': 'gradient',
            },
            'k1': {},
            'graph': {
                'sequential_edges': True,
                'semantic_edges': True,
                'semantic_threshold': 0.2,
                'random_shortcuts': 0.15,
            }
        },
        'model': {
            'hidden_dim': args.model_dim,  # Larger: 256 vs 96
            'n_layers': args.n_layers,     # More: 6 vs 4
            'n_heads': args.n_heads,       # More: 8 vs 4
            'dropout': 0.1,
        }
    }
    
    model = StateCorePipeline(config).to(device)
    return model, config


def create_baseline(args, vocab_size):
    """Create comparable baseline transformer."""
    model = BaselineTransformer(
        vocab_size=vocab_size,
        embed_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.model_dim * 4,
        max_seq_len=args.seq_length,
        dropout=0.1
    ).to(device)
    return model


def train_epoch(model, dataloader, optimizer, epoch, is_sosm=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        if is_sosm:
            logits, state = model(x)
        else:
            logits = model(x)
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.4f}, speed={speed:.1f} batch/s")
    
    return total_loss / total_batches


def evaluate(model, dataloader, is_sosm=True):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            if is_sosm:
                logits, _ = model(x)
            else:
                logits = model(x)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def generate_samples(model, tokenizer, prompts, max_tokens=50, is_sosm=True):
    """Generate text samples."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            
            for _ in range(max_tokens):
                ctx = input_ids[-64:]
                x = torch.tensor([ctx], dtype=torch.long, device=device)
                
                if is_sosm:
                    logits, _ = model(x)
                else:
                    logits = model(x)
                
                next_logits = logits[0, -1, :] / 0.3  # Low temperature
                
                # Repetition penalty
                for token in input_ids[-30:]:
                    if token < len(next_logits):
                        next_logits[token] /= 4.0
                
                # Top-k
                values, _ = torch.topk(next_logits, min(10, len(next_logits)))
                next_logits[next_logits < values[-1]] = float('-inf')
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                input_ids.append(next_token)
            
            results.append(tokenizer.decode(input_ids))
    
    return results


def main():
    args = get_args()
    
    print("=" * 70)
    print("SOSM LARGE-SCALE TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Model Dim: {args.model_dim}")
    print(f"  Embed Dim: {args.embed_dim}")
    print(f"  Time Dim: {args.time_dim}")
    print(f"  MU Layers: {args.mu_layers}")
    print(f"  Transformer Layers: {args.n_layers}")
    print(f"  Attention Heads: {args.n_heads}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print()
    
    # Load data
    print("Loading datasets...")
    train_dataset = MultiDomainDataset(
        split='train',
        seq_length=args.seq_length,
        max_samples_per_domain=5000 if args.quick else None
    )
    
    test_dataset = MultiDomainDataset(
        split='test',
        seq_length=args.seq_length,
        max_samples_per_domain=1000 if args.quick else None
    )
    
    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2)
    
    # Load tokenizer for generation
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    except:
        tokenizer = None
        print("Warning: Could not load tokenizer for generation")
    
    # ===== TRAIN SOSM =====
    print("\n" + "=" * 70)
    print("TRAINING SOSM")
    print("=" * 70)
    
    sosm, sosm_config = create_sosm(args, vocab_size)
    sosm_params = sum(p.numel() for p in sosm.parameters())
    print(f"SOSM Parameters: {sosm_params / 1e6:.1f}M")
    
    sosm_optimizer = torch.optim.AdamW(sosm.parameters(), lr=args.lr, weight_decay=0.01)
    sosm_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sosm_optimizer, T_max=args.epochs)
    
    best_sosm_ppl = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(sosm, train_loader, sosm_optimizer, epoch, is_sosm=True)
        test_loss, ppl = evaluate(sosm, test_loader, is_sosm=True)
        
        sosm_scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")
        
        # Save checkpoint
        if epoch % args.checkpoint_every == 0 or ppl < best_sosm_ppl:
            if ppl < best_sosm_ppl:
                best_sosm_ppl = ppl
                checkpoint_name = 'sosm_large_best.pt'
            else:
                checkpoint_name = f'sosm_large_epoch{epoch}.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': sosm.state_dict(),
                'optimizer_state_dict': sosm_optimizer.state_dict(),
                'config': sosm_config,
                'vocab_size': vocab_size,
                'test_loss': test_loss,
                'perplexity': ppl,
            }, checkpoint_name)
            print(f"  ✓ Saved: {checkpoint_name}")
        
        # Generate samples every 10 epochs
        if epoch % 10 == 0 and tokenizer:
            print("\n  Sample Generation:")
            prompts = ["The capital of India is", "Once upon a time"]
            samples = generate_samples(sosm, tokenizer, prompts, is_sosm=True)
            for prompt, sample in zip(prompts, samples):
                print(f"    '{prompt}' → {sample[:100]}...")
    
    print(f"\n✓ SOSM Training Complete. Best PPL: {best_sosm_ppl:.2f}")
    
    # ===== TRAIN BASELINE =====
    if not args.skip_baseline:
        print("\n" + "=" * 70)
        print("TRAINING BASELINE")
        print("=" * 70)
        
        baseline = create_baseline(args, vocab_size)
        baseline_params = sum(p.numel() for p in baseline.parameters())
        print(f"Baseline Parameters: {baseline_params / 1e6:.1f}M")
        
        baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=args.lr, weight_decay=0.01)
        baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optimizer, T_max=args.epochs)
        
        best_baseline_ppl = float('inf')
        
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            
            train_loss = train_epoch(baseline, train_loader, baseline_optimizer, epoch, is_sosm=False)
            test_loss, ppl = evaluate(baseline, test_loader, is_sosm=False)
            
            baseline_scheduler.step()
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Perplexity: {ppl:.2f}")
            
            if ppl < best_baseline_ppl:
                best_baseline_ppl = ppl
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': baseline.state_dict(),
                    'vocab_size': vocab_size,
                    'test_loss': test_loss,
                    'perplexity': ppl,
                }, 'baseline_large_best.pt')
                print(f"  ✓ Saved: baseline_large_best.pt")
        
        print(f"\n✓ Baseline Training Complete. Best PPL: {best_baseline_ppl:.2f}")
    
    # ===== FINAL COMPARISON =====
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"SOSM Best Perplexity: {best_sosm_ppl:.2f}")
    if not args.skip_baseline:
        print(f"Baseline Best Perplexity: {best_baseline_ppl:.2f}")
        improvement = (best_baseline_ppl - best_sosm_ppl) / best_baseline_ppl * 100
        print(f"SOSM Improvement: {improvement:.1f}%")
    
    # Final generation test
    if tokenizer:
        print("\n" + "-" * 70)
        print("FINAL GENERATION TEST")
        print("-" * 70)
        
        prompts = [
            "The capital of India is",
            "Once upon a time",
            "The future of AI",
            "def factorial(n):",
        ]
        
        print("\n📗 SOSM Outputs:")
        sosm_samples = generate_samples(sosm, tokenizer, prompts, is_sosm=True)
        for prompt, sample in zip(prompts, sosm_samples):
            print(f"  '{prompt}' → {sample[:150]}")
        
        if not args.skip_baseline:
            print("\n📘 Baseline Outputs:")
            baseline_samples = generate_samples(baseline, tokenizer, prompts, is_sosm=False)
            for prompt, sample in zip(prompts, baseline_samples):
                print(f"  '{prompt}' → {sample[:150]}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

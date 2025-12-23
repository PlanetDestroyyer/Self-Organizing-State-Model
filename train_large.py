#!/usr/bin/env python3
"""
SOSM Large-Scale Training Script

Trains SOSM with:
- Larger model dimensions (256 vs 96)
- More MU layers (4 vs 2)
- 50 epochs with checkpointing
- Built-in generation test

Usage:
    python train_large.py --epochs 50
    python train_large.py --epochs 10 --quick
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(description='SOSM Large-Scale Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=64, help='Sequence length')
    parser.add_argument('--model-dim', type=int, default=256, help='Model hidden dimension')
    parser.add_argument('--embed-dim', type=int, default=128, help='MU embedding dimension')
    parser.add_argument('--time-dim', type=int, default=64, help='TEMPORAL dimension')
    parser.add_argument('--mu-layers', type=int, default=4, help='Number of MU block-attention layers')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--checkpoint-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (smaller dataset)')
    return parser.parse_args()


def create_sosm(args, vocab_size):
    """Create SOSM with larger configuration."""
    config = {
        'stage': 3,
        'components': {
            'mu': {
                'vocab_size': vocab_size,
                'embed_dim': args.embed_dim,
                'max_seq_len': args.seq_length,
                'use_full_model': True,
                'mu_layers': args.mu_layers,
            },
            'temporal': {
                'time_dim': args.time_dim,
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
            'hidden_dim': args.model_dim,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'dropout': 0.1,
        }
    }
    
    model = StateCorePipeline(config).to(device)
    return model, config


def train_epoch(model, dataloader, optimizer, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, state = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            edges = state.routing_state.get('num_edges', 0) if state.routing_state else 0
            print(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.4f}, edges={edges}, speed={speed:.1f} batch/s")
    
    return total_loss / total_batches


def evaluate(model, dataloader):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def generate_samples(model, tokenizer, prompts, max_tokens=50):
    """Generate text samples."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            
            for _ in range(max_tokens):
                ctx = input_ids[-64:]
                x = torch.tensor([ctx], dtype=torch.long, device=device)
                logits, _ = model(x)
                
                next_logits = logits[0, -1, :] / 0.3
                
                # Strong repetition penalty
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
    print(f"Epochs: {args.epochs}")
    print(f"Model Dim: {args.model_dim}")
    print(f"Embed Dim: {args.embed_dim}")
    print(f"Time Dim: {args.time_dim}")
    print(f"MU Layers: {args.mu_layers}")
    print(f"Transformer Layers: {args.n_layers}")
    print(f"Attention Heads: {args.n_heads}")
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
    print(f"Vocab size: {vocab_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2)
    
    # Tokenizer for generation
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    except:
        tokenizer = None
    
    # Create model
    print("\nCreating SOSM...")
    model, config = create_sosm(args, vocab_size)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.1f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_ppl = float('inf')
    
    # Training loop
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        test_loss, ppl = evaluate(model, test_loader)
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")
        
        # Save checkpoint
        if epoch % args.checkpoint_every == 0 or ppl < best_ppl:
            if ppl < best_ppl:
                best_ppl = ppl
                name = 'sosm_large_best.pt'
            else:
                name = f'sosm_large_e{epoch}.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'vocab_size': vocab_size,
                'perplexity': ppl,
            }, name)
            print(f"  ✓ Saved: {name}")
        
        # Generate samples every 10 epochs
        if epoch % 10 == 0 and tokenizer:
            print("\n  Generation Test:")
            prompts = ["The capital of India is", "Once upon a time"]
            samples = generate_samples(model, tokenizer, prompts)
            for p, s in zip(prompts, samples):
                print(f"    '{p}' → {s[:80]}...")
    
    # Final test
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Perplexity: {best_ppl:.2f}")
    
    if tokenizer:
        print("\nFinal Generation:")
        prompts = ["The capital of India is", "The future of AI", "Once upon a time", "def factorial(n):"]
        samples = generate_samples(model, tokenizer, prompts)
        for p, s in zip(prompts, samples):
            print(f"  '{p}' → {s[:120]}")


if __name__ == '__main__':
    main()

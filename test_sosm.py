#!/usr/bin/env python3
"""
Test Self-Organizing State Model (SOSM)

Tests the full SOSM system with:
- MU semantic representations
- TEMPORAL time embeddings
- K-1 hierarchical credit assignment
- Graph-based attention routing

on WikiText, Code, and Scientific datasets.
Batch size: 64

Usage:
    python test_sosm.py
    python test_sosm.py --stage 0   # MU only
    python test_sosm.py --stage 1   # MU + TEMPORAL
    python test_sosm.py --stage 2   # MU + TEMPORAL + K-1
    python test_sosm.py --stage 3   # Full system
"""

import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Add TEMPORAL to path for proper imports
TEMPORAL_PATH = PROJECT_ROOT / "TEMPORAL"
if str(TEMPORAL_PATH) not in sys.path:
    sys.path.insert(0, str(TEMPORAL_PATH))

from sosm_data import create_dataloaders
from state_core import StateCorePipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Test SOSM')
    parser.add_argument('--stage', type=int, default=3,
                        help='Stage (0-3), default: 3 (full system)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    return parser.parse_args()


def train_epoch(pipeline, loader, optimizer, epoch: int, stage: int):
    """Train for one epoch."""
    pipeline.train()
    total_loss = 0
    total_nodes_updated = 0
    n_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (input_ids, labels, domains) in enumerate(loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward through SOSM
        logits, state = pipeline(input_ids)
        
        loss = F.cross_entropy(
            logits.view(-1, pipeline.vocab_size),
            labels.view(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
        
        # K-1 hierarchical updates (Stage 2+)
        if stage >= 2:
            responsibility = pipeline.backward_with_k1(loss.detach(), state, batch_idx)
            total_nodes_updated += responsibility.get('nodes_updated', 0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            log_str = f"  Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}"
            
            # Add stage-specific info
            if state.semantic_state is not None:
                log_str += f", semantic={state.semantic_state.shape[-1]}d"
            if state.temporal_state is not None:
                log_str += f", temporal={state.temporal_state.shape[-1]}d"
            if state.routing_state is not None:
                log_str += f", edges={state.routing_state.get('num_edges', 0)}"
            
            log_str += f", speed={batch_idx / max(1, elapsed):.1f} batch/s"
            print(log_str)
    
    avg_loss = total_loss / max(1, n_batches)
    avg_nodes = total_nodes_updated / max(1, n_batches) if stage >= 2 else 0
    
    return avg_loss, avg_nodes


def evaluate(pipeline, loader):
    """Evaluate model."""
    pipeline.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for input_ids, labels, domains in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits, state = pipeline(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, pipeline.vocab_size),
                labels.view(-1)
            )
            
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / max(1, n_batches)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def main():
    args = parse_args()
    
    print("=" * 70)
    print("SELF-ORGANIZING STATE MODEL (SOSM) TEST")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Stage: {args.stage}")
    print()
    
    # Stage descriptions
    stage_desc = {
        0: "MU only (baseline)",
        1: "MU + TEMPORAL",
        2: "MU + TEMPORAL + K-1",
        3: "Full system (MU + TEMPORAL + K-1 + Graph)"
    }
    print(f"Configuration: {stage_desc.get(args.stage, 'Unknown')}")
    print()
    
    # Config
    BATCH_SIZE = args.batch_size
    VOCAB_SIZE = 10000  # Smaller for testing
    SEQ_LEN = 64
    EPOCHS = args.epochs
    
    # Load data
    print("Loading datasets...")
    train_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        vocab_size=VOCAB_SIZE,
        seq_length=SEQ_LEN,
        domains=['wikitext', 'code', 'scientific']
    )
    print()
    
    # Create SOSM pipeline
    print("Creating SOSM pipeline...")
    config = {
        'stage': args.stage,
        'components': {
            'mu': {
                'vocab_size': VOCAB_SIZE,
                'embed_dim': 64,
                'max_seq_len': SEQ_LEN,
            },
            'temporal': {
                'time_dim': 32,
            },
            'k1': {},
            'graph': {
                'sequential_edges': True,
                'semantic_edges': args.stage == 3,
            }
        },
        'model': {
            'hidden_dim': 256,
            'n_layers': 4,
            'n_heads': 4,
            'dropout': 0.1,
        }
    }
    
    pipeline = StateCorePipeline(config).to(device)
    n_params = sum(p.numel() for p in pipeline.parameters())
    print(f"SOSM initialized: {n_params / 1e6:.1f}M parameters")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=1e-4)
    
    # Training loop
    print("Training...")
    print("-" * 70)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        train_loss, avg_nodes = train_epoch(
            pipeline, train_loader, optimizer, epoch, args.stage
        )
        test_loss, perplexity = evaluate(pipeline, test_loader)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        if args.stage >= 2:
            print(f"  Avg Nodes Updated (K-1): {avg_nodes:.1f}")
    
    print()
    print("=" * 70)
    print("SOSM TEST COMPLETE")
    print("=" * 70)
    
    # Save checkpoint with vocabulary
    checkpoint_name = f'sosm_stage{args.stage}_checkpoint.pt'
    char_to_idx, idx_to_char = train_loader.dataset.get_vocab()
    torch.save({
        'model_state_dict': pipeline.state_dict(),
        'stage': args.stage,
        'vocab_size': VOCAB_SIZE,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }, checkpoint_name)
    print(f"Saved: {checkpoint_name}")


if __name__ == '__main__':
    main()

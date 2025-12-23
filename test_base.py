#!/usr/bin/env python3
"""
Test Baseline Model

Tests a standard transformer baseline on WikiText, Code, and Scientific datasets.
Batch size: 64

Usage:
    python test_base.py
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sosm_data import create_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaselineTransformer(nn.Module):
    """
    Standard transformer baseline for comparison.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_dim: int = 1024,
        max_seq_len: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"Baseline initialized: {self.n_params / 1e6:.1f}M parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        
        # Embed
        h = self.token_emb(x) + self.pos_emb[:, :T]
        h = self.dropout(h)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        # Transform
        h = self.transformer(h, mask=mask)
        h = self.norm(h)
        
        # Output
        logits = self.output(h)
        return logits


def train_epoch(model, loader, optimizer, epoch: int):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (input_ids, labels, domains) in enumerate(loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            labels.view(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}, "
                  f"speed={batch_idx / max(1, elapsed):.1f} batch/s")
    
    return total_loss / max(1, n_batches)


def evaluate(model, loader):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for input_ids, labels, domains in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                labels.view(-1)
            )
            
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / max(1, n_batches)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def main():
    print("=" * 70)
    print("BASELINE TRANSFORMER TEST")
    print("=" * 70)
    print(f"Device: {device}")
    print()
    
    # Config
    BATCH_SIZE = 64
    VOCAB_SIZE = 10000  # Smaller for testing
    SEQ_LEN = 64
    EPOCHS = 3
    
    # Load data
    print("Loading datasets...")
    train_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        vocab_size=VOCAB_SIZE,
        seq_length=SEQ_LEN,
        domains=['wikitext', 'code', 'scientific']
    )
    print()
    
    # Create model
    print("Creating baseline model...")
    model = BaselineTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=256,
        n_layers=4,
        n_heads=4,
        ff_dim=512,
        max_seq_len=SEQ_LEN,
        dropout=0.1
    ).to(device)
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    print("Training...")
    print("-" * 70)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        test_loss, perplexity = evaluate(model, test_loader)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
    
    print()
    print("=" * 70)
    print("BASELINE TEST COMPLETE")
    print("=" * 70)
    
    # Save checkpoint with vocabulary
    char_to_idx, idx_to_char = train_loader.dataset.get_vocab()
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': VOCAB_SIZE,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }, 'baseline_checkpoint.pt')
    print("Saved: baseline_checkpoint.pt")


if __name__ == '__main__':
    main()

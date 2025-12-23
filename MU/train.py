"""
Standalone MU Transformer Training Script for Colab/Kaggle

This script trains both MU Transformer and Baseline Transformer,
then compares their performance on language modeling tasks.

Usage:
    python train.py
"""

# ============================================================================
# Installation (uncomment if running on Colab/Kaggle)
# ============================================================================
# !pip install torch transformers datasets tqdm matplotlib seaborn scikit-learn -q

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for both models"""
    # Model architecture
    vocab_size = 10000  # Smaller vocab for faster training
    max_seq_len = 128
    d_model = 128
    n_layers = 4  # Fewer layers for faster training
    n_heads = 4
    dropout = 0.1

    # MU specific
    r = 4  # MU matrix rows
    c = 4  # MU matrix columns

    # Training
    batch_size = 32
    num_epochs = 3  # Just a few epochs for demo
    learning_rate = 3e-4
    warmup_steps = 200
    max_train_steps = 1000  # Limit training for demo

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
print(f"Using device: {config.device}")

# ============================================================================
# DATA LOADING
# ============================================================================

class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration"""

    def __init__(self, split='train', max_seq_len=128, vocab_size=10000):
        from datasets import load_dataset

        print(f"Loading {split} dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Simple tokenization (character-level for demo)
        all_text = ' '.join([item['text'] for item in dataset if len(item['text'].strip()) > 0])

        # Build vocabulary
        if split == 'train':
            chars = sorted(list(set(all_text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars[:vocab_size-2])}
            self.char_to_idx['<PAD>'] = vocab_size - 2
            self.char_to_idx['<UNK>'] = vocab_size - 1

        # Tokenize
        self.data = []
        for i in range(0, len(all_text) - max_seq_len, max_seq_len // 2):
            chunk = all_text[i:i + max_seq_len + 1]
            if len(chunk) == max_seq_len + 1:
                tokens = [self.char_to_idx.get(c, vocab_size-1) for c in chunk]
                self.data.append(torch.tensor(tokens, dtype=torch.long))

        print(f"Created {len(self.data)} sequences")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return {
            'input_ids': seq[:-1],
            'labels': seq[1:]
        }

# ============================================================================
# MU TRANSFORMER IMPLEMENTATION
# ============================================================================

class MUAttentionLayer(nn.Module):
    """MU Attention Layer with semantic gating"""

    def __init__(self, r=4, c=4, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.r, self.c = r, c
        self.rc = r * c
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Projections
        self.W_q = nn.Linear(self.rc, d_model)
        self.W_k = nn.Linear(self.rc, d_model)
        self.W_v = nn.Linear(self.rc, d_model)
        self.W_out = nn.Linear(d_model, self.rc)
        self.W_g = nn.Linear(d_model, self.rc)
        self.W_b = nn.Linear(d_model, self.rc)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([r, c])

        # Sensitivity mask
        sensitivity_mask = torch.tensor([
            [0.1, 0.01, 0.01, 0.7],
            [0.7, 0.7, 0.7, 0.9],
            [0.9, 0.9, 0.9, 0.6],
            [0.6, 0.5, 0.5, 0.1]
        ], dtype=torch.float32)
        self.register_buffer('sensitivity_mask', sensitivity_mask)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.W_out, self.W_g, self.W_b]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, M, mask=None):
        B, T, r, c = M.shape
        M_flat = M.view(B, T, self.rc)

        # Project to Q, K, V
        Q = self.W_q(M_flat).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(M_flat).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(M_flat).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        context = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Project back to MU space
        delta_M = self.W_out(context).view(B, T, r, c)

        # Gating
        G = torch.sigmoid(self.W_g(context)).view(B, T, r, c)
        G = G * self.sensitivity_mask.unsqueeze(0).unsqueeze(0)

        # Bias
        B_term = torch.tanh(self.W_b(context)).view(B, T, r, c) * 0.1

        # Update
        M_updated = M * (1 - G) + delta_M * G + B_term
        M_updated = self.layer_norm(M_updated)

        return M_updated

class MUTransformer(nn.Module):
    """Complete MU Transformer"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.r, self.c = config.r, config.c
        self.rc = self.r * self.c

        # Token to MU embedding
        self.token_to_mu = nn.Embedding(config.vocab_size, self.rc)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, self.r, self.c) * 0.02)

        # MU layers
        self.layers = nn.ModuleList([
            MUAttentionLayer(self.r, self.c, config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Output
        self.output = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(self.rc, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.vocab_size)
        )

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).bool()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        B, T = input_ids.shape

        # Token to MU
        MU = self.token_to_mu(input_ids).view(B, T, self.r, self.c)
        MU = MU + self.pos_embedding[:, :T, :, :]
        MU = self.dropout(MU)

        # Causal mask
        mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)

        # Apply layers
        for layer in self.layers:
            MU = layer(MU, mask=mask)

        # Output
        logits = self.output(MU)

        return logits, MU

# ============================================================================
# BASELINE TRANSFORMER IMPLEMENTATION
# ============================================================================

class BaselineTransformer(nn.Module):
    """Standard Transformer for comparison"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)

        # Output
        self.output = nn.Linear(config.d_model, config.vocab_size)

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len) * float('-inf'), diagonal=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape

        # Embeddings
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout(x)

        # Transformer
        mask = self.causal_mask[:T, :T]
        x = self.transformer(x, mask=mask, is_causal=True)

        # Output
        logits = self.output(x)

        return logits, None

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, max_steps=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for i, batch in enumerate(pbar):
        if max_steps and i >= max_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward
        logits, MU = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches

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
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(min(avg_loss, 20))

    return {'loss': avg_loss, 'perplexity': perplexity}

def get_scheduler(optimizer, warmup_steps, total_steps):
    """Cosine warmup scheduler"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 70)
    print("MU TRANSFORMER vs BASELINE TRANSFORMER")
    print("=" * 70)
    print()

    # Load data
    print("Loading datasets...")
    try:
        train_dataset = SimpleTextDataset('train', config.max_seq_len, config.vocab_size)
        val_dataset = SimpleTextDataset('validation', config.max_seq_len, config.vocab_size)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Using dummy data for demonstration...")

        # Create dummy data
        class DummyDataset(Dataset):
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                seq_len = config.max_seq_len
                return {
                    'input_ids': torch.randint(0, config.vocab_size, (seq_len,)),
                    'labels': torch.randint(0, config.vocab_size, (seq_len,))
                }

        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    print()

    # Results storage
    results = {
        'MU Transformer': {'train_loss': [], 'val_loss': [], 'val_ppl': []},
        'Baseline': {'train_loss': [], 'val_loss': [], 'val_ppl': []}
    }

    # Train both models
    for model_name, ModelClass in [('MU Transformer', MUTransformer), ('Baseline', BaselineTransformer)]:
        print("=" * 70)
        print(f"Training {model_name}")
        print("=" * 70)
        print()

        # Create model
        model = ModelClass(config).to(config.device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params:,}")
        print()

        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * config.num_epochs
        scheduler = get_scheduler(optimizer, config.warmup_steps, total_steps)

        # Training loop
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                config.device, max_steps=config.max_train_steps
            )

            # Evaluate
            val_metrics = evaluate(model, val_loader, config.device)

            # Store results
            results[model_name]['train_loss'].append(train_loss)
            results[model_name]['val_loss'].append(val_metrics['loss'])
            results[model_name]['val_ppl'].append(val_metrics['perplexity'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Perplexity: {val_metrics['perplexity']:.2f}")

        print()

    # ========================================================================
    # RESULTS COMPARISON
    # ========================================================================

    print("=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)
    print()

    print(f"{'Model':<20} {'Final Val Loss':<15} {'Final Perplexity':<15}")
    print("-" * 50)
    for model_name in ['MU Transformer', 'Baseline']:
        val_loss = results[model_name]['val_loss'][-1]
        val_ppl = results[model_name]['val_ppl'][-1]
        print(f"{model_name:<20} {val_loss:<15.4f} {val_ppl:<15.2f}")

    print()

    # Compute improvement
    mu_ppl = results['MU Transformer']['val_ppl'][-1]
    baseline_ppl = results['Baseline']['val_ppl'][-1]

    if mu_ppl < baseline_ppl:
        improvement = ((baseline_ppl - mu_ppl) / baseline_ppl) * 100
        print(f"✓ MU Transformer is {improvement:.1f}% better in perplexity")
    else:
        diff = ((mu_ppl - baseline_ppl) / baseline_ppl) * 100
        print(f"✗ MU Transformer is {diff:.1f}% worse in perplexity")

    print()

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    print("Generating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss comparison
    epochs = range(1, config.num_epochs + 1)
    axes[0].plot(epochs, results['MU Transformer']['val_loss'], 'o-', label='MU Transformer', linewidth=2)
    axes[0].plot(epochs, results['Baseline']['val_loss'], 's-', label='Baseline', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Validation Loss Comparison', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Perplexity comparison
    axes[1].plot(epochs, results['MU Transformer']['val_ppl'], 'o-', label='MU Transformer', linewidth=2)
    axes[1].plot(epochs, results['Baseline']['val_ppl'], 's-', label='Baseline', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].set_title('Perplexity Comparison', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mu_transformer_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'mu_transformer_comparison.png'")
    plt.show()

    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == '__main__':
    main()

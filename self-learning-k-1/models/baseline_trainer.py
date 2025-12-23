"""
FAIR Baseline Trainer: Same Transformer Architecture as K-1, Dense Updates

This uses the EXACT same ModularSparseTransformer architecture but with
standard dense backpropagation (update ALL parameters every step).
This enables fair comparison with K-1's sparse updates.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaselineTrainer:
    """
    Baseline trainer using the SAME architecture as K-1 but with DENSE updates.
    
    Key difference from K-1:
    - K-1: Updates only top-k parameter groups per step (sparse)
    - Baseline: Updates ALL parameters every step (dense/standard backprop)
    """
    
    def __init__(self, config: Dict, data_loader=None):
        self.config = config
        self.data_loader = data_loader
        
        # Get vocab size
        if data_loader is not None:
            vocab_size = data_loader.get_vocab_size()
        else:
            vocab_size = config.get('model', {}).get('vocab_size', 10000)
        
        self.vocab_size = vocab_size
        embed_dim = config.get('model', {}).get('embed_dim', 128)
        ff_dim = config.get('model', {}).get('hidden_dim', 256)
        num_heads = config.get('model', {}).get('num_heads', 4)
        num_layers = config.get('model', {}).get('num_layers', 4)
        max_seq_len = config.get('model', {}).get('max_seq_len', 64)
        
        # Use SAME architecture as K-1 for fair comparison
        from k1_system.core.modular_transformer import ModularSparseTransformer
        self.model = ModularSparseTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            dropout=0.1
        ).to(device)
        
        # Single optimizer for ALL parameters (dense updates)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.get('learning', {}).get('learning_rate', 0.001),
            weight_decay=0.01
        )
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.model.parameters())
        
        # Config
        self.seq_length = max_seq_len
        self.log_interval = config.get('learning', {}).get('log_interval', 5000)
        self.validation_interval = config.get('learning', {}).get('validation_interval', 10000)
        
        # Statistics
        self.total_params_updated = 0
        self.total_steps = 0
        self.loss_history = []

        # AMP for speed
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        
        print(f"\n" + "="*70)
        print("BASELINE: Same Architecture, Dense Updates")
        print("="*70)
        print(f"Total parameters: {self.total_params:,}")
        print(f"Update strategy: ALL parameters every step")
        print("="*70 + "\n")
    
    def train(self, data=None, max_steps: int = 1000):
        """Train with standard dense backpropagation."""
        print("\n" + "="*70)
        print("BASELINE: Dense Backpropagation (Update ALL)")
        print("="*70)
        print(f"Device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total parameters: {self.total_params:,}")
        print(f"Batch size: {self.config.get('learning', {}).get('batch_size', 32)}")
        print("="*70 + "\n")
        
        self.model.train()
        start_time = time.time()
        loss_fn = nn.CrossEntropyLoss()
        batch_size = self.config.get('learning', {}).get('batch_size', 32)
        
        for step in range(max_steps):
            # Get batch
            if self.data_loader:
                try:
                    x_tokens, y_tokens = self.data_loader.get_batch('train', batch_size=batch_size, return_tensors='pt')
                except:
                    x_tokens = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
                    y_tokens = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
            else:
                x_tokens = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
                y_tokens = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=device)
            
            # Forward with AMP
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = self.model(x_tokens)
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, self.vocab_size),
                    y_tokens[:, 1:].reshape(-1)
                )
            
            # Backward with scaler
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Stats
            self.loss_history.append(loss.item())
            self.total_params_updated += self.total_params
            self.total_steps += 1
            
            # Logging
            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                
                gpu_mem = ""
                if device.type == 'cuda':
                    mem = torch.cuda.memory_allocated() / 1e9
                    gpu_mem = f" | GPU: {mem:.2f}GB"
                
                print(f"[{step:6d}] Loss: {loss.item():.4f} | "
                      f"Params: {self.total_params:,} (100%) | "
                      f"Speed: {steps_per_sec:.1f} step/s{gpu_mem}")
            
            # Validation
            if step % self.validation_interval == 0 and self.data_loader:
                val_loss, val_ppl = self._validate()
                print(f"[{step:6d}] VALIDATION | Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print("Baseline Training Complete")
        print("="*70)
        print(f"Total steps: {self.total_steps}")
        print(f"Time: {elapsed:.1f}s")
        print("="*70)
        
        return {
            'time': elapsed,
            'steps': self.total_steps,
            'final_loss': self.loss_history[-1] if self.loss_history else 0
        }
    
    def _validate(self):
        """Run validation."""
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            x, y = self.data_loader.get_batch('val', batch_size=32, return_tensors='pt')
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = self.model(x)
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, self.vocab_size),
                    y[:, 1:].reshape(-1)
                )
        
        self.model.train()
        return loss.item(), np.exp(loss.item())

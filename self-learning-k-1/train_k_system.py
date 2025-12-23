#!/usr/bin/env python3
"""
K-1 Self-Learning System — Complete Training Wrapper

This is the main entry point to train the K-1 system.

Phases:
- Phase 1 (0 to N steps): Human-controlled, fixed parameters
- Phase 2 (N+ steps): Self-controlled, staged autonomy

Usage:
    python train_k_system.py                    # Default training
    python train_k_system.py --phase1-steps 50000  # Custom Phase 1 duration
    python train_k_system.py --dataset wikitext    # Specific dataset
"""

import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader
from k1_system.core import K1_Tree
from k1_system.training import HierarchicalK1Trainer
from k1_system.autonomy import Phase2Controller, Action, BoundarySystem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='K-1 Self-Learning System Training')
    parser.add_argument('--phase1-steps', type=int, default=10000,
                        help='Steps for Phase 1 (human-controlled), default: 10000')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum steps (None = let system decide in Phase 2)')
    parser.add_argument('--dataset', type=str, default='wikitext',
                        choices=['wikitext', 'code_python', 'scientific'],
                        help='Dataset to train on')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--log-interval', type=int, default=500,
                        help='Logging interval')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file')
    return parser.parse_args()


def load_config(config_path: str = None) -> dict:
    """Load configuration file."""
    if config_path:
        with open(config_path) as f:
            return json.load(f)
    
    # Default config
    default_path = Path(__file__).parent / 'k1_system' / 'config' / 'config_phase1.json'
    if default_path.exists():
        with open(default_path) as f:
            return json.load(f)
    
    # Fallback config
    return {
        'model': {
            'embed_dim': 128,
            'hidden_dim': 256,
            'num_heads': 4,
            'tree_depth': 4,
            'branching_factor': [4, 3, 2],
            'max_seq_len': 64,
            'vocab_size': 10000
        },
        'learning': {
            'learning_rate': 0.001,
            'batch_size': 256,
            'top_k': 5,
            'log_interval': 500
        },
        'training': {
            'max_steps': 100000
        }
    }


class K1SystemTrainer:
    """
    Complete K-1 training system with Phase 1 → Phase 2 transition.
    
    Phase 1: Human-controlled training with fixed parameters
    Phase 2: Self-controlled training with staged autonomy
    """
    
    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        
        # Override config with args
        config['learning']['batch_size'] = args.batch_size
        config['learning']['log_interval'] = args.log_interval
        
        # Load data
        print(f"\n📦 Loading dataset: {args.dataset}")
        self.data_loader = DataLoader(
            dataset_name=args.dataset,
            vocab_size=config['model'].get('vocab_size', 10000),
            seq_length=config['model'].get('max_seq_len', 64)
        )
        print(f"   Vocabulary size: {self.data_loader.get_vocab_size()}")
        
        # Create trainer
        print("\n🌳 Initializing K-1 Hierarchical System...")
        self.trainer = HierarchicalK1Trainer(config, self.data_loader)
        
        # Phase 2 controller
        self.phase_controller = Phase2Controller(phase_1_steps=args.phase1_steps)
        
        # Training state
        self.step = 0
        self.total_loss = 0.0
        self.start_time = None
        
        # Max steps (None = let Phase 2 decide)
        self.max_steps = args.max_steps
        
        print(f"\n⚙️ Configuration:")
        print(f"   Phase 1 duration: {args.phase1_steps} steps")
        print(f"   Max steps: {self.max_steps or 'Phase 2 will decide'}")
        print(f"   Batch size: {args.batch_size}")
    
    def train(self):
        """Main training loop with Phase 1 → Phase 2 transition."""
        print("\n" + "=" * 70)
        print("🚀 K-1 SELF-LEARNING SYSTEM — Training Started")
        print("=" * 70)
        print(f"Device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 70 + "\n")
        
        self.start_time = time.time()
        loss_fn = nn.CrossEntropyLoss()
        batch_size = self.args.batch_size
        
        # Training loop
        while True:
            self.step += 1
            
            # Check stopping conditions
            if self.max_steps and self.step > self.max_steps:
                print(f"\n✅ Reached max steps ({self.max_steps})")
                break
            
            # Phase 2 self-stopping
            should_stop, reason = self.phase_controller.should_stop()
            if should_stop:
                print(f"\n🛑 SELF-STOPPING: {reason}")
                print("   System decided training is complete!")
                break
            
            # Get batch
            try:
                x, y = self.data_loader.get_batch('train', batch_size=batch_size, return_tensors='pt')
            except Exception as e:
                x = torch.randint(0, self.trainer.vocab_size, (batch_size, self.trainer.seq_length), device=device)
                y = torch.randint(0, self.trainer.vocab_size, (batch_size, self.trainer.seq_length), device=device)
            
            # Forward pass
            self.trainer.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, path = self.trainer.model(x)
                loss = loss_fn(
                    logits[:, :-1].reshape(-1, self.trainer.vocab_size),
                    y[:, 1:].reshape(-1)
                )
            
            # Backward pass
            self.trainer.scaler.scale(loss).backward()
            self.trainer.scaler.unscale_(self.trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(self.trainer.model.parameters(), max_norm=1.0)
            
            # Hierarchical error attribution (new fast method)
            self.trainer.model.fast_hierarchical_step(loss, self.step)
            
            # Optimizer step
            self.trainer.scaler.step(self.trainer.optimizer)
            self.trainer.scaler.update()
            
            # Update phase controller
            current_phase = self.phase_controller.step(loss=loss.item())
            
            # Accumulate loss
            self.total_loss += loss.detach()
            
            # Logging
            if self.step % self.args.log_interval == 0:
                self._log_progress(loss.item())
        
        # Training complete
        self._training_complete()
    
    def _log_progress(self, current_loss: float):
        """Log training progress."""
        avg_loss = (self.total_loss / self.step).item()
        elapsed = time.time() - self.start_time
        speed = self.step / elapsed if elapsed > 0 else 0
        
        phase_str = f"Phase {self.phase_controller.phase}"
        if self.phase_controller.phase == 2:
            stage = self.phase_controller.boundary_system.current_stage_num
            phase_str += f" (Stage {stage})"
        
        print(f"\n[{self.step:6d}] Loss: {avg_loss:.4f} | {phase_str} | Speed: {speed:.1f} step/s")
        print("─" * 60)
        
        # Show error path from fast_hierarchical_step
        if hasattr(self.trainer.model, '_error_path'):
            path = self.trainer.model._error_path
            grads = self.trainer.model._path_gradients
            path_str = " → ".join(f"Node{idx}(g={grads[i]:.2f})" for i, idx in enumerate(path))
            print(f"Error Path: {path_str}")
        
        # Summary
        nodes_updated = len(getattr(self.trainer.model, '_error_path', []))
        total_nodes = len(self.trainer.model.all_nodes)
        print(f"Updated: {nodes_updated}/{total_nodes} nodes ({nodes_updated/total_nodes*100:.0f}%)")
        
        # Phase 2 status
        if self.phase_controller.phase == 2:
            status = self.phase_controller.boundary_system.get_status()
            print(f"Autonomy: {status['successful_cheats']}/{status['cheats_to_advance']} cheats to next stage")
    
    def _training_complete(self):
        """Finalize training and save checkpoint."""
        elapsed = time.time() - self.start_time
        avg_loss = (self.total_loss / self.step).item()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        print(f"   Total steps: {self.step}")
        print(f"   Final loss: {avg_loss:.4f}")
        print(f"   Total time: {elapsed/60:.1f} minutes")
        print(f"   Final phase: {self.phase_controller.phase}")
        
        if self.phase_controller.phase == 2:
            status = self.phase_controller.boundary_system.get_status()
            print(f"   Autonomy stage: {status['stage']} ({status['stage_name']})")
            print(f"   Total cheats: {status['total_cheats']}")
        
        # Save checkpoint
        checkpoint_path = Path(__file__).parent / 'checkpoints' / 'k1_system.pt'
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'step': self.step,
            'loss': avg_loss,
            'phase': self.phase_controller.phase,
            'config': self.config,
            'vocab': {
                'word_to_idx': self.data_loader.word_to_idx,
                'idx_to_word': self.data_loader.idx_to_word,
            }
        }
        
        if self.phase_controller.phase == 2:
            checkpoint['autonomy_status'] = self.phase_controller.boundary_system.get_status()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")
        print("=" * 70)


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create trainer and run
    trainer = K1SystemTrainer(config, args)
    trainer.train()


if __name__ == '__main__':
    main()

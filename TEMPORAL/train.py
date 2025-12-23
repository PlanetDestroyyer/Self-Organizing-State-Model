"""
Production-Grade Training Script for TEMPORAL
SOTA datasets, mixed precision, proper logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_scheduler
from datasets import load_dataset
import os
import json
from tqdm import tqdm
import numpy as np
import argparse

from config import get_config, print_config
from model import create_model, verify_gradient_flow, Temporal_Transformer

# Optional: Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# DATA LOADING (SOTA)
# ============================================================================

def load_and_prepare_dataset(config):
    """
    Load SOTA dataset (WikiText-103, The Pile, C4, etc.)
    """
    print(f"\nLoading dataset: {config.dataset_name} ({config.dataset_config})")

    try:
        # Load dataset
        if config.dataset_name == "wikitext":
            dataset = load_dataset(config.dataset_name, config.dataset_config)
        elif config.dataset_name == "pile":
            dataset = load_dataset("EleutherAI/pile", streaming=True)
        elif config.dataset_name == "c4":
            dataset = load_dataset("allenai/c4", "en", streaming=True)
        else:
            # Custom dataset from HuggingFace
            if config.dataset_config:
                dataset = load_dataset(config.dataset_name, config.dataset_config)
            else:
                dataset = load_dataset(config.dataset_name)

        print(f"✓ Dataset loaded successfully")

        # Check if dataset has validation split, if not create one
        if 'validation' not in dataset:
            print("⚠️  No validation split found, creating from train data (90/10 split)...")
            train_val_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
            dataset['train'] = train_val_split['train']
            dataset['validation'] = train_val_split['test']
            print(f"✓ Created validation split: train={len(dataset['train']):,}, val={len(dataset['validation']):,}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✓ Tokenizer loaded: {config.tokenizer_name}")
        print(f"  Vocabulary size: {len(tokenizer)}")

        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                max_length=config.block_size,
                padding='max_length',
                return_tensors='pt'
            )

        print("✓ Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            num_proc=config.preprocessing_num_workers
        )

        # Apply limits for debugging
        if config.max_train_samples:
            tokenized_dataset['train'] = tokenized_dataset['train'].select(
                range(min(config.max_train_samples, len(tokenized_dataset['train'])))
            )

        if config.max_eval_samples and 'validation' in tokenized_dataset:
            tokenized_dataset['validation'] = tokenized_dataset['validation'].select(
                range(min(config.max_eval_samples, len(tokenized_dataset['validation'])))
            )

        print(f"✓ Training samples: {len(tokenized_dataset['train']):,}")
        if 'validation' in tokenized_dataset:
            print(f"✓ Validation samples: {len(tokenized_dataset['validation']):,}")

        return tokenized_dataset, tokenizer

    except Exception as e:
        print(f"\n⚠️  Failed to load {config.dataset_name}: {e}")
        print("Generating synthetic data for testing...")
        return generate_synthetic_dataset(config)


def generate_synthetic_dataset(config):
    """Fallback: Generate synthetic data if dataset loading fails"""
    from torch.utils.data import Dataset

    class SyntheticDataset(Dataset):
        def __init__(self, num_samples, seq_len, vocab_size):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            return {'input_ids': input_ids, 'labels': input_ids.clone()}

    train_dataset = SyntheticDataset(10000, config.block_size, config.vocab_size)
    val_dataset = SyntheticDataset(1000, config.block_size, config.vocab_size)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    dataset_dict = {
        'train': train_dataset,
        'validation': val_dataset
    }

    print(f"✓ Generated synthetic dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    return dataset_dict, tokenizer


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Production-grade trainer with mixed precision, gradient accumulation, etc."""

    def __init__(self, model, config, train_dataset, eval_dataset, tokenizer):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.use_cpu else "cpu")
        print(f"\nUsing device: {self.device}")

        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model.to(self.device)

        # Custom collate function to handle tokenized data
        def collate_fn(batch):
            """Collate function for DataLoader"""
            if isinstance(batch[0], dict):
                # Stack all tensors in the batch
                return {
                    key: torch.stack([torch.tensor(item[key]) if not isinstance(item[key], torch.Tensor) else item[key] for item in batch])
                    for key in batch[0].keys()
                }
            else:
                return torch.utils.data.dataloader.default_collate(batch)

        # DataLoaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            collate_fn=collate_fn
        )

        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.eval_batch_size,
                num_workers=config.dataloader_num_workers,
                pin_memory=config.dataloader_pin_memory,
                collate_fn=collate_fn
            )
        else:
            self.eval_dataloader = None

        # Optimizer (AdamW is SOTA for transformers)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        num_training_steps = len(self.train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        num_warmup_steps = int(config.warmup_ratio * num_training_steps) if config.warmup_steps is None else config.warmup_steps

        self.lr_scheduler = get_scheduler(
            config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Mixed precision
        self.use_amp = config.bf16 or config.fp16
        self.scaler = GradScaler() if (config.fp16 and self.device.type == "cuda") else None

        # Tracking
        self.global_step = 0
        self.best_eval_loss = float('inf')

        # Weights & Biases
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            # Login with API key if provided (avoids interactive prompt)
            if hasattr(config, 'wandb_api_key') and config.wandb_api_key:
                wandb.login(key=config.wandb_api_key)

            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.run_name,
                config=vars(config)
            )

        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.logging_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)

        # Verify gradient flow for TEMPORAL model
        is_temporal = isinstance(self.model, Temporal_Transformer)
        if is_temporal:
            verify_gradient_flow(self.model)

        # Store epoch results for logging
        epoch_results = []

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            train_loss = self.train_epoch(epoch)

            # Evaluation
            if self.eval_dataloader:
                eval_loss = self.evaluate()
                perplexity = np.exp(eval_loss)
                print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Eval Loss={eval_loss:.4f}, Perplexity={perplexity:.2f}")

                # Store results
                epoch_results.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                    'perplexity': perplexity
                })

                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint('best')
            else:
                print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}")
                epoch_results.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss
                })

        # Final save
        self.save_checkpoint('final')

        # Log final results to output.txt
        model_type = "TEMPORAL" if is_temporal else "BASELINE"
        with open("output.txt", 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"{model_type} MODEL TRAINING RESULTS (Seed: {self.config.seed})\n")
            f.write(f"{'='*70}\n\n")

            for result in epoch_results:
                f.write(f"Epoch {result['epoch']}: ")
                f.write(f"Train Loss={result['train_loss']:.4f}")
                if 'eval_loss' in result:
                    f.write(f", Eval Loss={result['eval_loss']:.4f}")
                    f.write(f", Perplexity={result['perplexity']:.2f}")
                f.write("\n")

            # Write final summary
            if epoch_results:
                final = epoch_results[-1]
                f.write(f"\nFinal Results:\n")
                f.write(f"  Train Loss: {final['train_loss']:.4f}\n")
                if 'eval_loss' in final:
                    f.write(f"  Eval Loss: {final['eval_loss']:.4f}\n")
                    f.write(f"  Perplexity: {final['perplexity']:.2f}\n")
            f.write("\n")

        if self.use_wandb:
            wandb.finish()

        print("\n✅ Training complete!")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = input_ids.clone()

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp, dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                # Only pass update_time to TEMPORAL models (Baseline doesn't support it)
                if isinstance(self.model, Temporal_Transformer):
                    outputs = self.model(input_ids, labels=labels, update_time=True)
                else:
                    outputs = self.model(input_ids, labels=labels)
                loss = outputs['loss'] / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.lr_scheduler.get_last_lr()[0]
                    if self.use_wandb:
                        wandb.log({
                            'train/loss': loss.item() * self.config.gradient_accumulation_steps,
                            'train/lr': lr,
                            'train/step': self.global_step,
                            'train/epoch': epoch
                        })

                # Evaluation
                if self.config.eval_strategy == "steps" and self.global_step % self.config.eval_steps == 0:
                    if self.eval_dataloader:
                        eval_loss = self.evaluate()
                        print(f"\nStep {self.global_step}: Eval Loss={eval_loss:.4f}, PPL={np.exp(eval_loss):.2f}")
                        self.model.train()

                # Checkpointing
                if self.config.save_strategy == "steps" and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f'step_{self.global_step}')

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item() * self.config.gradient_accumulation_steps})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = input_ids.clone()

            # Only pass update_time to TEMPORAL models (Baseline doesn't support it)
            if isinstance(self.model, Temporal_Transformer):
                outputs = self.model(input_ids, labels=labels, update_time=False)
            else:
                outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if self.use_wandb:
            wandb.log({
                'eval/loss': avg_loss,
                'eval/perplexity': np.exp(avg_loss),
                'eval/step': self.global_step
            })

        return avg_loss

    def save_checkpoint(self, name):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'{name}.pt')

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'config': vars(self.config)
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

        if self.use_wandb:
            wandb.save(checkpoint_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TEMPORAL model")
    parser.add_argument('--config', type=str, default='colab',
                        choices=['production', 'colab', 'scaled', 'debug'],
                        help='Configuration preset (colab: fast validation, scaled: comprehensive)')
    parser.add_argument('--model-type', type=str, default='temporal',
                        choices=['temporal', 'baseline'],
                        help='Model type to train')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config seed)')
    args = parser.parse_args()

    # Load config
    config = get_config(args.config)

    # Override seed if provided
    if args.seed is not None:
        config.seed = args.seed

    print_config(config)
    print(f"\n🎲 Using random seed: {config.seed}\n")

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Make cudnn deterministic (may be slower)
    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load dataset
    dataset_dict, tokenizer = load_and_prepare_dataset(config)

    # Create model
    model = create_model(config, args.model_type)

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict.get('validation'),
        tokenizer=tokenizer
    )

    trainer.train()


if __name__ == "__main__":
    main()

"""
Phase 2.9: SOSM Training with Word-Level Tokenization
======================================================

Key Changes from Phase 2.7:
- Uses word-level tokenizer instead of BPE
- Vocabulary: ~100k words vs 50k BPE tokens
- Each token = complete semantic unit (aligns with MU blocks)
- Input corruption now operates on meaningful words
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import yaml
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from word_tokenizer import WordLevelTokenizer
from state_core import StateCorePipeline
from multi_dataset_loader import load_simple_wikipedia
from generation_prompts import WIKI_PROMPTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Import corruption functions from train_robust_generation.py
from train_robust_generation import (
    get_inverse_sigmoid_schedule,
    generate_stability_test
)


def corrupt_word_tokens(input_ids, epsilon, vocab_size, pad_token_id):
    """
    Corrupt word-level tokens (same logic as BPE version).
    Now operates on complete words instead of fragments.
    """
    if epsilon == 0.0:
        return input_ids
    
    corruption_mask = torch.bernoulli(
        torch.full(input_ids.shape, epsilon, device=input_ids.device)
    ).bool()
    
    padding_mask = (input_ids == pad_token_id)
    corruption_mask = corruption_mask & ~padding_mask
    
    noise_tokens = torch.randint(
        0, vocab_size,
        input_ids.shape,
        device=input_ids.device
    )
    
    corrupted = torch.where(corruption_mask, noise_tokens, input_ids)
    return corrupted


class WordLevelDataset(torch.utils.data.Dataset):
    """Dataset for word-level tokenized texts."""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input/labels
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Pad
        pad_len = self.max_length - 1 - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def train_epoch_word_level(model, train_loader, optimizer, scaler, global_step, 
                            k_schedule=500, vocab_size=100000):
    """Train one epoch with word-level tokens and corruption."""
    model.train()
    total_loss = 0
    total_clean_loss = 0
    n_batches = 0
    epoch_corruptions = []
    
    for batch in tqdm(train_loader, desc="Training (Word-Level)"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Calculate corruption
        corruption_ratio = get_inverse_sigmoid_schedule(global_step, k=k_schedule)
        epoch_corruptions.append(corruption_ratio)
        
        # Corrupt words (not fragments!)
        corrupted_input = corrupt_word_tokens(
            input_ids,
            epsilon=corruption_ratio,
            vocab_size=vocab_size,
            pad_token_id=0  # PAD token
        )
        
        optimizer.zero_grad()
        
        with autocast():
            logits, state = model(corrupted_input, return_state=True)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            if state is not None and hasattr(state, 'reg_losses'):
                loss = loss + state.reg_losses['total_reg']
        
        # Track clean loss
        with torch.no_grad():
            clean_logits, _ = model(input_ids, return_state=True)
            clean_loss = F.cross_entropy(
                clean_logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_clean_loss += clean_loss.item()
        n_batches += 1
        global_step += 1
    
    return {
        'loss': total_loss / n_batches,
        'clean_loss': total_clean_loss / n_batches,
        'corruption_ratio': np.mean(epoch_corruptions),
        'delta_bias': (total_loss - total_clean_loss) / n_batches,
        'global_step': global_step
    }


def evaluate_word_level(model, test_loader, vocab_size):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids, return_state=True)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def train_word_level_sosm(epochs=10, batch_size=64, max_samples=50000, k_schedule=500):
    """
    Train SOSM with word-level tokenization.
    """
    print("="*70)
    print("PHASE 2.9: WORD-LEVEL SOSM TRAINING")
    print("="*70)
    print("Tokenization: Word-Level (semantic units)")
    print(f"Corruption Schedule: Inverse Sigmoid (k={k_schedule})")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print()
    
    # Load word tokenizer
    print("Loading word-level tokenizer...")
    tokenizer = WordLevelTokenizer()
    tokenizer.load('word_tokenizer_vocab.json')
    vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size: {vocab_size}")
    
    # Load data
    print("\nLoading Simple Wikipedia...")
    texts = load_simple_wikipedia(max_samples=max_samples)
    split = int(len(texts) * 0.9)
    
    train_dataset = WordLevelDataset(texts[:split], tokenizer, max_length=128)
    test_dataset = WordLevelDataset(texts[split:], tokenizer, max_length=128)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Create SOSM with new vocab size
    print("\nInitializing SOSM...")
    with open('configs/tier1.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update vocab size in the correct nested location
    config['components']['mu']['vocab_size'] = vocab_size
    
    model = StateCorePipeline(config).to(device)
    print(f"  SOSM params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Configured vocab size: {vocab_size}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = GradScaler()
    
    # Training
    results = {
        'config': {'epochs': epochs, 'k_schedule': k_schedule, 'vocab_size': vocab_size},
        'training_history': [],
        'generation_samples': []
    }
    
    global_step = 0
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")
        
        train_metrics = train_epoch_word_level(
            model, train_loader, optimizer, scaler, global_step, k_schedule, vocab_size
        )
        global_step = train_metrics['global_step']
        
        test_loss, test_ppl = evaluate_word_level(model, test_loader, vocab_size)
        
        # Generation test
        print(f"\n📝 Generation Test (Epoch {epoch+1}):")
        test_prompt = WIKI_PROMPTS[0]
        gen_result = generate_stability_test(model, tokenizer, test_prompt)
        
        print(f"  Prompt: {test_prompt}")
        print(f"  Generated: {gen_result['text'][:200]}...")
        print(f"  Unique Ratio: {gen_result['unique_ratio']:.3f}")
        
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_clean_loss': train_metrics['clean_loss'],
            'corruption_ratio': train_metrics['corruption_ratio'],
            'delta_bias': train_metrics['delta_bias'],
            'test_loss': test_loss,
            'test_ppl': test_ppl,
            'generation': gen_result
        }
        results['training_history'].append(epoch_results)
        
        print(f"\nMetrics:")
        print(f"  Train Loss (Corrupted): {train_metrics['loss']:.4f}")
        print(f"  Train Loss (Clean): {train_metrics['clean_loss']:.4f}")
        print(f"  Δ_Bias: {train_metrics['delta_bias']:.4f}")
        print(f"  Test PPL: {test_ppl:.2f}")
        print(f"  Corruption: {train_metrics['corruption_ratio']:.1%}")
    
    # Final generation
    print(f"\n{'='*70}")
    print("FINAL GENERATION SAMPLES")
    print(f"{'='*70}\n")
    
    for i, prompt in enumerate(WIKI_PROMPTS[:3]):
        gen_result = generate_stability_test(model, tokenizer, prompt, max_length=100)
        results['generation_samples'].append(gen_result)
        
        print(f"Sample {i+1}:")
        print(f"  Prompt: {prompt}")
        print(f"  Generated: {gen_result['text']}")
        print(f"  Unique: {gen_result['unique_ratio']:.2%}")
        print("-" * 70)
    
    # Save
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'results': results
    }, output_dir / 'sosm_word_level_phase2_9.pt')
    
    with open(output_dir / 'phase2_9_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Phase 2.9 Complete!")
    print(f"   Model: results/sosm_word_level_phase2_9.pt")
    print(f"   Results: results/phase2_9_results.json")
    
    return results


if __name__ == '__main__':
    results = train_word_level_sosm(
        epochs=10,
        batch_size=32,  # Reduced from 64 due to larger vocab (188M params)
        max_samples=50000,
        k_schedule=500
    )
    
    print("\n" + "="*70)
    print("PHASE 2.9 SUMMARY")
    print("="*70)
    final = results['training_history'][-1]
    final_gen = results['generation_samples'][-1]
    
    print(f"Final Test PPL: {final['test_ppl']:.2f}")
    print(f"Final Δ_Bias: {final['delta_bias']:.4f}")
    print(f"Generation Unique: {final_gen['unique_ratio']:.2%}")
    print(f"Generation Length: {final_gen['num_tokens']} tokens")
    
    print("\nSuccess Criteria:")
    print(f"  ✓ PPL < 5.0: {final['test_ppl']:.2f} {'✅' if final['test_ppl'] < 5.0 else '❌'}")
    print(f"  ✓ Δ_Bias < 0.5: {final['delta_bias']:.4f} {'✅' if final['delta_bias'] < 0.5 else '❌'}")
    print(f"  ✓ Coherence: Check generation samples above")

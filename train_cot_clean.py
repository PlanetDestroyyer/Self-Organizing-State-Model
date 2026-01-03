"""
Phase 2.8b: SOSM Training with Clean Chain-of-Thought
======================================================

Key Changes from Phase 2.8:
- NO INPUT CORRUPTION (corruption was fighting CoT learning)
- Pure CoT training on clean, structured reasoning examples
- Fixes Δ_Bias by removing corruption conflict
- Expects Δ_Bias → 0.0, improved coherence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import yaml
import json
import time
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from word_tokenizer import WordLevelTokenizer
from cot_data_generator import CoTDataGenerator
from state_core import StateCorePipeline
from multi_dataset_loader import load_simple_wikipedia
from generation_prompts import WIKI_PROMPTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Removed corruption functions - training on clean CoT data only


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


class CoTDataset(torch.utils.data.Dataset):
    """Dataset for CoT-augmented word-level tokenized texts."""
    
    def __init__(self, texts, tokenizer, cot_generator=None, max_length=128, cot_ratio=0.6):
        self.texts = texts
        self.tokenizer = tokenizer
        self.cot_generator = cot_generator or CoTDataGenerator()
        self.max_length = max_length
        self.cot_ratio = cot_ratio
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Add CoT with probability cot_ratio
        if random.random() < self.cot_ratio and self.cot_generator:
            text = self.cot_generator.generate_cot_mixed(text)
        
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


def train_epoch_cot_clean(model, train_loader, optimizer, scaler, vocab_size=100000):
    """Train one epoch with clean CoT data (no corruption)."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in tqdm(train_loader, desc="Training (CoT Clean)"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            logits, state = model(input_ids, return_state=True)  # Clean input
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            if state is not None and hasattr(state, 'reg_losses'):
                loss = loss + state.reg_losses['total_reg']
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'global_step': n_batches
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


def generate_word_level(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.95):
    """Generate text using word-level tokenizer."""
    model.eval()
    
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        generated = input_ids.clone()
        
        for _ in range(max_length):
            if generated.shape[1] >= 512:
                break
            
            logits, _ = model(generated, return_state=True)
            next_token_logits = logits[:, -1, :]
            
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum <= top_p
            mask[..., 0] = True
            filtered_probs = sorted_probs * mask.float()
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            next_token_idx = torch.multinomial(filtered_probs, 1)
            next_token = torch.gather(sorted_idx, -1, next_token_idx)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        tokens = generated[0].tolist()
        
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        unique_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        repetitions = 0
        for i in range(len(tokens) - 6):
            if tokens[i:i+3] == tokens[i+3:i+6]:
                repetitions += 1
        repetition_rate = repetitions / max(1, len(tokens) - 6)
        
        return {
            'text': text,
            'length': len(text),
            'num_tokens': total_tokens,
            'unique_ratio': unique_ratio,
            'repetition_rate': repetition_rate
        }


def train_cot_clean(epochs=10, batch_size=32, max_samples=50000, cot_ratio=0.7):
    """
    Train SOSM with Clean Chain-of-Thought (no corruption).
    """
    print("="*70)
    print("PHASE 2.8b: CLEAN CHAIN-OF-THOUGHT SOSM TRAINING")
    print("="*70)
    print("Tokenization: Word-Level (semantic units)")
    print("CoT Augmentation: Enabled")
    print(f"CoT Ratio: {cot_ratio:.0%}")
    print("Corruption: DISABLED (clean training)")
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
    
    train_dataset = CoTDataset(texts[:split], tokenizer, max_length=128, cot_ratio=cot_ratio)
    test_dataset = CoTDataset(texts[split:], tokenizer, max_length=128, cot_ratio=cot_ratio)
    
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
        'config': {'epochs': epochs, 'cot_ratio': cot_ratio, 'vocab_size': vocab_size},
        'training_history': [],
        'generation_samples': []
    }
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")
        
        train_metrics = train_epoch_cot_clean(
            model, train_loader, optimizer, scaler, vocab_size
        )
        
        test_loss, test_ppl = evaluate_word_level(model, test_loader, vocab_size)
        
        # Generation test
        print(f"\n📝 Generation Test (Epoch {epoch+1}):")
        test_prompt = WIKI_PROMPTS[0]
        gen_result = generate_word_level(model, tokenizer, test_prompt)
        
        print(f"  Prompt: {test_prompt}")
        print(f"  Generated: {gen_result['text'][:200]}...")
        print(f"  Unique Ratio: {gen_result['unique_ratio']:.3f}")
        
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'test_loss': test_loss,
            'test_ppl': test_ppl,
            'generation': gen_result
        }
        results['training_history'].append(epoch_results)
        
        print(f"\nMetrics:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Test PPL: {test_ppl:.2f}")
    
    # Final generation
    print(f"\n{'='*70}")
    print("FINAL GENERATION SAMPLES")
    print(f"{'='*70}\n")
    
    for i, prompt in enumerate(WIKI_PROMPTS[:3]):
        gen_result = generate_word_level(model, tokenizer, prompt, max_length=100)
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
    }, output_dir / 'sosm_cot_clean_phase2_8b.pt')
    
    with open(output_dir / 'phase2_8b_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Phase 2.8b Complete!")
    print(f"   Model: results/sosm_cot_clean_phase2_8b.pt")
    print(f"   Results: results/phase2_8b_results.json")
    
    return results


if __name__ == '__main__':
    results = train_cot_clean(
        epochs=10,
        batch_size=32,
        max_samples=50000,
        cot_ratio=0.7  # 70% of data gets CoT augmentation
    )
    
    print("="*70)
    print("PHASE 2.8b SUMMARY")
    print("="*70)
    final = results['training_history'][-1]
    final_gen = results['generation_samples'][-1]
    
    print(f"Final Test PPL: {final['test_ppl']:.2f}")
    print(f"Generation Unique: {final_gen['unique_ratio']:.2%}")
    print(f"Generation Length: {final_gen['num_tokens']} tokens")
    
    print("\nSuccess Criteria:")
    print(f"  ✓ PPL < 5.0: {final['test_ppl']:.2f} {'✅' if final['test_ppl'] < 5.0 else '❌'}")
    print(f"  ✓ Coherence: Check generation samples above")

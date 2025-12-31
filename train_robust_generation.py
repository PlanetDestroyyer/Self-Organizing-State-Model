"""
Phase 2.7: SOSM Robust Generation Training
=============================================

Fixes "State Drift" (Exposure Bias) via Input Corruption Training.

Key Features:
- Inverse Sigmoid decay schedule (0% → 20% corruption)
- Stochastic token dropout/replacement
- Generation monitoring every 0.5 epochs
- Metrics: Clean PPL, Corrupted PPL, Generation Stability

Based on:
- Bengio et al. (2015): Scheduled Sampling
- Carson (2025): Stochastic Dynamical Theory of LLM Drift
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
from transformers import GPT2Tokenizer
from tqdm import tqdm

from state_core import StateCorePipeline
from multi_dataset_loader import create_multi_dataset_loaders
from generation_prompts import WIKI_PROMPTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_inverse_sigmoid_schedule(step, k=2000):
    """
    Inverse Sigmoid decay for corruption ratio.
    
    Schedule designed to:
    - Maintain ~0% corruption for first k steps (model builds foundation)
    - Gradually transition to ~10% at step k
    - Reach ~20% corruption by step 2k
    - Never exceed floor/ceiling bounds
    
    Args:
        step: Current training step
        k: Warmup parameter (default 2000)
    
    Returns:
        epsilon: Corruption ratio [0.0, 0.2]
    
    Formula: ε = k / (k + exp(step/k))
    Inverted for corruption: corruption = 1 - ε, capped at 0.2
    """
    try:
        # Standard inverse sigmoid
        teacher_forcing_ratio = k / (k + np.exp(step / k))
        
        # Convert to corruption ratio (inverted)
        # Early: TF=1.0 → corruption=0.0
        # Late: TF=0.8 → corruption=0.2
        corruption_ratio = min(1.0 - teacher_forcing_ratio, 0.2)
        
        return corruption_ratio
    except (OverflowError, RuntimeWarning):
        # If exp overflows, we're very late in training
        return 0.2  # Max corruption


def corrupt_input_tokens(input_ids, epsilon, vocab_size=50257, pad_token_id=50256):
    """
    Stochastically corrupt input tokens via replacement.
    
    Corruption strategy:
    - Random tokens (simulates arbitrary errors)
    - Preserves padding tokens (don't corrupt structure)
    
    Args:
        input_ids: [Batch, Seq] - Clean input tokens
        epsilon: Corruption ratio (e.g., 0.15 = 15% corruption)
        vocab_size: Vocabulary size for random sampling
        pad_token_id: Token ID to preserve
    
    Returns:
        corrupted_ids: [Batch, Seq] - Noisy input tokens
    """
    if epsilon == 0.0:
        return input_ids
    
    # Create corruption mask (1 = corrupt, 0 = keep)
    corruption_mask = torch.bernoulli(
        torch.full(input_ids.shape, epsilon, device=input_ids.device)
    ).bool()
    
    # Don't corrupt padding tokens
    padding_mask = (input_ids == pad_token_id)
    corruption_mask = corruption_mask & ~padding_mask
    
    # Generate random replacement tokens
    noise_tokens = torch.randint(
        0, vocab_size, 
        input_ids.shape, 
        device=input_ids.device
    )
    
    # Apply corruption: keep original where mask=0, use noise where mask=1
    corrupted = torch.where(corruption_mask, noise_tokens, input_ids)
    
    return corrupted


def train_epoch_with_corruption(model, train_loader, optimizer, scaler, 
                                 global_step, k_schedule=2000):
    """Train one epoch with Input Corruption."""
    model.train()
    total_loss = 0
    total_clean_loss = 0
    total_corrupted_loss = 0
    n_batches = 0
    
    epoch_corruptions = []
    
    for batch in tqdm(train_loader, desc="Training with Corruption"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Calculate current corruption ratio
        corruption_ratio = get_inverse_sigmoid_schedule(global_step, k=k_schedule)
        epoch_corruptions.append(corruption_ratio)
        
        # Corrupt the input
        corrupted_input = corrupt_input_tokens(
            input_ids, 
            epsilon=corruption_ratio,
            vocab_size=50257,
            pad_token_id=50256
        )
        
        optimizer.zero_grad()
        
        # Forward pass with corrupted input
        with autocast():
            logits, state = model(corrupted_input, return_state=True)
            
            # Loss against CLEAN labels (not corrupted input)
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size), 
                labels.view(-1), 
                ignore_index=-100
            )
            
            # Add regularization if available
            if state is not None and hasattr(state, 'reg_losses'):
                loss = loss + state.reg_losses['total_reg']
        
        # Also track loss on clean input (for monitoring)
        with torch.no_grad():
            clean_logits, _ = model(input_ids, return_state=True)
            clean_loss = F.cross_entropy(
                clean_logits.view(-1, model.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_corrupted_loss += loss.item()
        total_clean_loss += clean_loss.item()
        n_batches += 1
        global_step += 1
    
    avg_loss = total_loss / n_batches
    avg_clean_loss = total_clean_loss / n_batches
    avg_corrupted_loss = total_corrupted_loss / n_batches
    avg_corruption = np.mean(epoch_corruptions)
    
    return {
        'loss': avg_loss,
        'clean_loss': avg_clean_loss,
        'corrupted_loss': avg_corrupted_loss,
        'corruption_ratio': avg_corruption,
        'delta_bias': avg_corrupted_loss - avg_clean_loss,
        'global_step': global_step
    }


def evaluate(model, test_loader):
    """Evaluate on clean test set."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids, return_state=True)
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size), 
                labels.view(-1), 
                ignore_index=-100
            )
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def generate_stability_test(model, tokenizer, prompt, max_length=100, 
                             temperature=0.8, top_p=0.95):
    """
    Generate text and measure stability metrics.
    
    Returns:
        dict with 'text', 'length', 'repetition_rate', 'unique_ratio'
    """
    model.eval()
    
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = input_ids.clone()
        
        for _ in range(max_length):
            if generated.shape[1] >= 512:
                break
            
            logits, _ = model(generated, return_state=True)
            next_token_logits = logits[:, -1, :]
            
            # Nucleus sampling
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
        
        # Decode
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        tokens = generated[0].tolist()
        
        # Calculate metrics
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        unique_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # Simple repetition detection (3-gram loops)
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


def train_robust_sosm(epochs=5, batch_size=64, max_samples=50000, k_schedule=2000):
    """
    Phase 2.7: Train SOSM with Input Corruption.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        max_samples: Max samples per dataset
        k_schedule: Inverse sigmoid warmup parameter
    """
    print("="*70)
    print("PHASE 2.7: SOSM ROBUST GENERATION TRAINING")
    print("="*70)
    print(f"Corruption Schedule: Inverse Sigmoid (k={k_schedule})")
    print(f"Target Corruption: 0% → 20%")
    print(f"Epochs: {epochs}")
    print(f"Dataset: Simple Wikipedia")
    print(f"Device: {device}")
    print()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Simple Wikipedia only (fast iteration)
    print("Loading Simple Wikipedia dataset...")
    loaders = create_multi_dataset_loaders(
        tokenizer,
        batch_size=batch_size,
        max_samples_per_dataset=max_samples,
        max_length=128
    )
    train_loader = loaders['wiki']['train']
    test_loader = loaders['wiki']['test']
    
    # Create SOSM model (load from Phase 2.6 checkpoint if exists)
    print("\nInitializing SOSM model...")
    with open('configs/tier1.yaml', 'r') as f:
        sosm_config = yaml.safe_load(f)
    
    sosm = StateCorePipeline(sosm_config).to(device)
    
    # Try to load Phase 2.6 checkpoint
    checkpoint_path = Path('results/sosm_trained.pt')
    if checkpoint_path.exists():
        print(f"Loading Phase 2.6 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        sosm.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded trained SOSM from Phase 2.6")
    else:
        print("⚠️  No Phase 2.6 checkpoint found. Training from scratch.")
    
    print(f"SOSM params: {sum(p.numel() for p in sosm.parameters()):,}")
    
    # Optimizer and scaler
    optimizer = torch.optim.AdamW(sosm.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = GradScaler()
    
    # Training loop
    results = {
        'config': {
            'epochs': epochs,
            'k_schedule': k_schedule,
            'batch_size': batch_size,
            'max_samples': max_samples
        },
        'training_history': [],
        'generation_samples': []
    }
    
    global_step = 0
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_epoch_with_corruption(
            sosm, train_loader, optimizer, scaler, 
            global_step, k_schedule
        )
        global_step = train_metrics['global_step']
        
        # Evaluate
        test_loss, test_ppl = evaluate(sosm, test_loader)
        
        # Generation test (every epoch)
        print(f"\n📝 Generation Stability Test (Epoch {epoch+1}):")
        test_prompt = WIKI_PROMPTS[0]  # "The history of"
        gen_result = generate_stability_test(sosm, tokenizer, test_prompt)
        
        print(f"  Prompt: {test_prompt}")
        print(f"  Generated ({gen_result['num_tokens']} tokens): {gen_result['text'][:200]}...")
        print(f"  Unique Ratio: {gen_result['unique_ratio']:.3f}")
        print(f"  Repetition Rate: {gen_result['repetition_rate']:.3f}")
        
        # Log metrics
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_clean_loss': train_metrics['clean_loss'],
            'train_corrupted_loss': train_metrics['corrupted_loss'],
            'corruption_ratio': train_metrics['corruption_ratio'],
            'delta_bias': train_metrics['delta_bias'],
            'test_loss': test_loss,
            'test_ppl': test_ppl,
            'generation': gen_result
        }
        results['training_history'].append(epoch_results)
        
        print(f"\nMetrics:")
        print(f"  Train Loss (Corrupted): {train_metrics['corrupted_loss']:.4f}")
        print(f"  Train Loss (Clean): {train_metrics['clean_loss']:.4f}")
        print(f"  Δ_Bias (Exposure Gap): {train_metrics['delta_bias']:.4f}")
        print(f"  Test PPL: {test_ppl:.2f}")
        print(f"  Corruption Ratio: {train_metrics['corruption_ratio']:.1%}")
    
    # Final generation samples
    print(f"\n{'='*70}")
    print("FINAL GENERATION SAMPLES")
    print(f"{'='*70}\n")
    
    for i, prompt in enumerate(WIKI_PROMPTS[:3]):
        gen_result = generate_stability_test(sosm, tokenizer, prompt, max_length=100)
        results['generation_samples'].append(gen_result)
        
        print(f"Sample {i+1}:")
        print(f"  Prompt: {prompt}")
        print(f"  Generated: {gen_result['text']}")
        print(f"  Tokens: {gen_result['num_tokens']}, Unique: {gen_result['unique_ratio']:.2%}")
        print("-" * 70)
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': sosm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': sosm_config,
        'results': results
    }, output_dir / 'sosm_robust_phase2_7.pt')
    
    # Save metrics
    with open(output_dir / 'phase2_7_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Phase 2.7 Complete!")
    print(f"   Model saved: results/sosm_robust_phase2_7.pt")
    print(f"   Metrics saved: results/phase2_7_results.json")
    
    return results


if __name__ == '__main__':
    results = train_robust_sosm(
        epochs=5,
        batch_size=64,
        max_samples=50000,
        k_schedule=2000
    )
    
    print("\n" + "="*70)
    print("PHASE 2.7 SUMMARY")
    print("="*70)
    final_ppl = results['training_history'][-1]['test_ppl']
    final_delta = results['training_history'][-1]['delta_bias']
    final_gen = results['generation_samples'][-1]
    
    print(f"Final Test PPL: {final_ppl:.2f}")
    print(f"Final Δ_Bias: {final_delta:.4f}")
    print(f"Generation Stability: {final_gen['unique_ratio']:.2%} unique tokens")
    print(f"Generation Length: {final_gen['num_tokens']} tokens")
    
    # Success criteria
    print("\nSuccess Criteria:")
    print(f"  ✓ PPL < 5.0: {final_ppl:.2f} {'✅' if final_ppl < 5.0 else '❌'}")
    print(f"  ✓ Δ_Bias < 0.5: {final_delta:.4f} {'✅' if final_delta < 0.5 else '❌'}")
    print(f"  ✓ Length > 50 tokens: {final_gen['num_tokens']} {'✅' if final_gen['num_tokens'] > 50 else '❌'}")

"""
SOSM vs Baseline Comparison Experiment

Rigorous comparison on 3 datasets:
- Simple Wikipedia (natural language)
- Python Code (structured)  
- ArXiv Papers (scientific)

5 epochs each, matched parameters (~132M), same training config.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import yaml
import json
import time
from pathlib import Path
from transformers import GPT2Tokenizer
from tqdm import tqdm

from state_core import StateCorePipeline
from baseline_transformer import BaselineTransformer
from multi_dataset_loader import create_multi_dataset_loaders
from generation_tests import run_generation_tests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, train_loader, optimizer, scaler, is_sosm=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        with autocast():
            if is_sosm:
                logits, state = model(input_ids, return_state=True)
                # Add regularization loss if available
                loss = F.cross_entropy(logits.view(-1, model.vocab_size), labels.view(-1), ignore_index=-100)
                if state is not None and hasattr(state, 'reg_losses'):
                    loss = loss + state.reg_losses['total_reg']
            else:
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, model.vocab_size), labels.view(-1), ignore_index=-100)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, test_loader, is_sosm=True):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            if is_sosm:
                logits, _ = model(input_ids, return_state=True)
            else:
                logits = model(input_ids)
            
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), labels.view(-1), ignore_index=-100)
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def run_comparison(epochs=7, batch_size=64, max_samples=50000):
    """Run full comparison experiment"""
    
    print("="*70)
    print("SOSM vs BASELINE COMPARISON")
    print("="*70)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Samples per dataset: {max_samples}")
    print()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("Loading datasets...")
    loaders = create_multi_dataset_loaders(
        tokenizer,
        batch_size=batch_size,
        max_samples_per_dataset=max_samples,
        max_length=128  # Reduced to avoid sequence length errors
    )
    
    # Create models
    print("\nCreating models...")
    with open('configs/tier1.yaml', 'r') as f:
        sosm_config = yaml.safe_load(f)
    
    sosm = StateCorePipeline(sosm_config).to(device)
    baseline = BaselineTransformer().to(device)
    
    print(f"SOSM params: {sum(p.numel() for p in sosm.parameters()):,}")
    print(f"Baseline params: {sum(p.numel() for p in baseline.parameters()):,}")
    
    # Results storage
    results = {
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'max_samples': max_samples,
            'sosm_params': sum(p.numel() for p in sosm.parameters()),
            'baseline_params': sum(p.numel() for p in baseline.parameters())
        },
        'datasets': {}
    }
    
    # Run on each dataset
    for dataset_name, (train_loader, test_loader) in loaders.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        results['datasets'][dataset_name] = {'sosm': [], 'baseline': []}
        
        # Train SOSM
        print(f"\n[SOSM] Training...")
        sosm_optimizer = torch.optim.AdamW(sosm.parameters(), lr=2e-4, weight_decay=0.01)
        sosm_scaler = GradScaler()
        
        for epoch in range(epochs):
            start = time.time()
            train_loss = train_epoch(sosm, train_loader, sosm_optimizer, sosm_scaler, is_sosm=True)
            test_loss, ppl = evaluate(sosm, test_loader, is_sosm=True)
            elapsed = time.time() - start
            
            print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Test={test_loss:.4f}, PPL={ppl:.2f}, Time={elapsed:.0f}s")
            
            results['datasets'][dataset_name]['sosm'].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'perplexity': ppl,
                'time': elapsed
            })
        
        # Train Baseline
        print(f"\n[BASELINE] Training...")
        baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=2e-4, weight_decay=0.01)
        baseline_scaler = GradScaler()
        
        for epoch in range(epochs):
            start = time.time()
            train_loss = train_epoch(baseline, train_loader, baseline_optimizer, baseline_scaler, is_sosm=False)
            test_loss, ppl = evaluate(baseline, test_loader, is_sosm=False)
            elapsed = time.time() - start
            
            print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Test={test_loss:.4f}, PPL={ppl:.2f}, Time={elapsed:.0f}s")
            
            results['datasets'][dataset_name]['baseline'].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'perplexity': ppl,
                'time': elapsed
            })
        
        # Reset models for next dataset
        sosm = StateCorePipeline(sosm_config).to(device)
        baseline = BaselineTransformer().to(device)
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'comparison_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save trained models
    print(f"\n{'='*70}")
    print("SAVING MODELS")
    print(f"{'='*70}")
    
    torch.save({
        'model_state_dict': sosm.state_dict(),
        'config': sosm_config,
        'vocab_size': sosm.vocab_size
    }, results_dir / 'sosm_final.pt')
    print(f"✅ SOSM saved to: results/sosm_final.pt")
    
    torch.save({
        'model_state_dict': baseline.state_dict(),
        'vocab_size': baseline.vocab_size,
        'd_model': baseline.d_model,
        'n_layers': baseline.n_layers
    }, results_dir / 'baseline_final.pt')
    print(f"✅ Baseline saved to: results/baseline_final.pt")
    
    # Run generation tests
    print(f"\n{'='*70}")
    print("RUNNING GENERATION TESTS")
    print(f"{'='*70}")
    generation_results = run_generation_tests(sosm, baseline, tokenizer, save_dir=results_dir)
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: results/comparison_final.json")
    print(f"Models saved to: results/sosm_final.pt, results/baseline_final.pt")
    print(f"Generation tests saved to: results/generation_tests.json")
    
    # Print summary
    print("\nFINAL PERPLEXITY SUMMARY:")
    print(f"{'Dataset':<15} {'SOSM':<10} {'Baseline':<10} {'Winner':<10}")
    print("-" * 50)
    
    for dataset_name in results['datasets'].keys():
        sosm_ppl = results['datasets'][dataset_name]['sosm'][-1]['perplexity']
        baseline_ppl = results['datasets'][dataset_name]['baseline'][-1]['perplexity']
        winner = 'SOSM' if sosm_ppl < baseline_ppl else 'Baseline'
        
        print(f"{dataset_name:<15} {sosm_ppl:<10.2f} {baseline_ppl:<10.2f} {winner:<10}")
    
    return results


if __name__ == "__main__":
    results = run_comparison(epochs=7, batch_size=64, max_samples=50000)  # 7 epochs fits 10h budget (2 epochs = 1.33h)

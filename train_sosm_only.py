"""
SOSM-Only Training Script

Train SOSM on 3 datasets for maximum epochs (fits 10h budget).
Run this on Device 1 while baseline runs on Device 2.
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
from multi_dataset_loader import create_multi_dataset_loaders
from generation_prompts import WIKI_PROMPTS, CODE_PROMPTS, ARXIV_PROMPTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, train_loader, optimizer, scaler):
    """Train SOSM for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        with autocast():
            logits, state = model(input_ids, return_state=True)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), labels.view(-1), ignore_index=-100)
            if state is not None and hasattr(state, 'reg_losses'):
                loss = loss + state.reg_losses['total_reg']
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, test_loader):
    """Evaluate SOSM on test set"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids, return_state=True)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), labels.view(-1), ignore_index=-100)
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def generate_examples(model, tokenizer, prompts, domain_name, max_length=200, num_examples=12):
    """Generate text examples for qualitative evaluation"""
    model.eval()
    results = []
    
    print(f"\n{'='*70}")
    print(f"GENERATING {domain_name.upper()} EXAMPLES (SOSM)")
    print(f"{'='*70}\n")
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts[:num_examples]):
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            # Generate
            generated = input_ids.clone()
            for _ in range(max_length):
                if generated.shape[1] >= 512:  # Max sequence length
                    break
                    
                logits, _ = model(generated, return_state=True)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop at EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # Decode
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            print(f"Example {i+1}:")
            print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
            print(f"Generated: {generated_text[:500]}..." if len(generated_text) > 500 else f"Generated: {generated_text}")
            print("-" * 70)
            
            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'length': len(generated_text)
            })
    
    return results


def train_sosm(epochs=15, batch_size=64, max_samples=50000):
    """Train SOSM only (for parallel training with baseline on other device)"""
    
    print("="*70)
    print("SOSM TRAINING (Device 1)")
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
        max_length=128
    )
    
    # Create SOSM model
    print("\nCreating SOSM model...")
    with open('configs/tier1.yaml', 'r') as f:
        sosm_config = yaml.safe_load(f)
    
    sosm = StateCorePipeline(sosm_config).to(device)
    print(f"SOSM params: {sum(p.numel() for p in sosm.parameters()):,}")
    
    # Results storage
    results = {
        'config': {
            'model': 'SOSM',
            'epochs': epochs,
            'batch_size': batch_size,
            'max_samples': max_samples,
            'params': sum(p.numel() for p in sosm.parameters())
        },
        'datasets': {}
    }
    
    # Train on each dataset
    for dataset_name, (train_loader, test_loader) in loaders.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        results['datasets'][dataset_name] = []
        
        optimizer = torch.optim.AdamW(sosm.parameters(), lr=2e-4, weight_decay=0.01)
        scaler = GradScaler()
        
        for epoch in range(epochs):
            start = time.time()
            train_loss = train_epoch(sosm, train_loader, optimizer, scaler)
            test_loss, ppl = evaluate(sosm, test_loader)
            elapsed = time.time() - start
            
            print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Test={test_loss:.4f}, PPL={ppl:.2f}, Time={elapsed:.0f}s")
            
            results['datasets'][dataset_name].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'perplexity': ppl,
                'time': elapsed
            })
        
        # Generate examples after training on this dataset
        print(f"\n{'='*70}")
        print(f"GENERATING EXAMPLES FOR {dataset_name.upper()}")
        print(f"{'='*70}")
        
        if dataset_name == 'simple_wiki':
            gen_results = generate_examples(sosm, tokenizer, WIKI_PROMPTS, "Wikipedia Text")
        elif dataset_name == 'code':
            gen_results = generate_examples(sosm, tokenizer, CODE_PROMPTS, "Python Code")
        elif dataset_name == 'arxiv':
            gen_results = generate_examples(sosm, tokenizer, ARXIV_PROMPTS, "ArXiv Articles")
        
        results['datasets'][dataset_name].append({
            'generation_examples': gen_results
        })
        
        # Reset model for next dataset
        sosm = StateCorePipeline(sosm_config).to(device)
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'sosm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save final model
    torch.save({
        'model_state_dict': sosm.state_dict(),
        'config': sosm_config,
        'vocab_size': sosm.vocab_size
    }, results_dir / 'sosm_trained.pt')
    
    print(f"\n{'='*70}")
    print("SOSM TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: results/sosm_results.json")
    print(f"Model saved to: results/sosm_trained.pt")
    
    # Print summary
    print("\nFINAL PERPLEXITY:")
    for dataset_name in results['datasets'].keys():
        final_ppl = results['datasets'][dataset_name][-1]['perplexity']
        print(f"  {dataset_name}: {final_ppl:.2f}")
    
    return results


if __name__ == "__main__":
    # 15 epochs fits in 10 hours (2 epochs = 1.33h × 3 datasets = 4h)
    # 15 epochs × 3 datasets = ~10 hours
    results = train_sosm(epochs=15, batch_size=64, max_samples=50000)

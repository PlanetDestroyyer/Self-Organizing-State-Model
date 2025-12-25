"""
Manual K Hyperparameter Study - Phase 2.3.3

Automated experiments to find optimal K value for semantic edges.

Tests K values: [3, 5, 7, 10, 12, 15]
For each K:
- Train 5 epochs
- Measure: PPL, edge count, homonym separation, speed
- Save results

Output: Recommendation for optimal K based on PPL/speed tradeoff.

Usage:
    python experiments/k_study.py --epochs 5 --device cuda
"""

import sys
from pathlib import Path
import torch
import time
import json
from datetime import datetime

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from state_core.pipeline import StateCorePipeline
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import argparse


def load_wikitext_data(tokenizer, max_length=128, batch_size=8):
    """Load WikiText-2 dataset."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
    
    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"✓ Loaded {len(dataset)} examples, {len(dataloader)} batches")
    return dataloader


def test_homonym_separation(model, tokenizer, device):
    """
    Test if model can distinguish homonyms based on context.
    
    Returns separation score (higher = better context understanding).
    """
    homonym_tests = [
        ("The baseball bat flew through the air.", "bat", "sports"),
        ("The bat hung upside down in the cave.", "bat", "animal"),
        ("I deposited money in the bank.", "bank", "finance"),
        ("We sat by the river bank.", "bank", "nature"),
    ]
    
    separations = []
    
    for text, word, context in homonym_tests:
        # Tokenize
        tokens = tokenizer.encode(text, return_tensors='pt').to(device)
        
        # Find word position
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        word_pos = None
        for i in range(len(tokens[0]) - len(word_tokens) + 1):
            if all(tokens[0][i+j] == word_tokens[j] for j in range(len(word_tokens))):
                word_pos = i
                break
        
        if word_pos is None:
            continue
        
        # Get MU state at word position
        with torch.no_grad():
            _, state = model(tokens)
            mu_state = state.semantic_state[0, word_pos].cpu()
        
        separations.append(mu_state)
    
    # Compute average separation between contexts
    if len(separations) >= 2:
        total_sep = 0
        count = 0
        for i in range(0, len(separations), 2):
            if i+1 < len(separations):
                sep = (separations[i] - separations[i+1]).norm().item()
                total_sep += sep
                count += 1
        
        return total_sep / count if count > 0 else 0.0
    
    return 0.0


def train_with_k(k_value, epochs=5, device='cuda', max_steps=1000):
    """
    Train model with specific K value and measure metrics.
    
    Returns dict with results.
    """
    print(f"\n{'='*60}")
    print(f"Testing K = {k_value}")
    print(f"{'='*60}")
    
    # Create config with this K
    config = {
        'model': {'hidden_dim': 512, 'n_heads': 8},
        'mu': {
            'use_full_model': False,
            'use_contextual_refinement': False,  # Keep simple for K study
        },
        'temporal': {},
        'pipeline': {
            'graph': {
                'enabled': True,
                'semantic_edges': True,
                'semantic_k': k_value,  # THIS IS WHAT WE'RE TESTING
                'semantic_threshold': 0.05,
                'semantic_method': 'topk',
                'streaming_topk': True
            }
        }
    }
    
    # Initialize model
    print(f"Initializing model with K={k_value}...")
    model = StateCorePipeline(config).to(device)
    model.train()
    
    # Initialize tokenizer and data
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataloader = load_wikitext_data(tokenizer, max_length=128, batch_size=8)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training metrics
    results = {
        'k': k_value,
        'epochs': epochs,
        'ppls': [],
        'edge_counts': [],
        'speeds': [],
        'final_ppl': None,
        'avg_edges': None,
        'avg_speed': None,
        'homonym_separation': None
    }
    
    # Training loop
    total_steps = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_steps = 0
        epoch_edges = []
        
        for batch_idx, batch in enumerate(dataloader):
            if total_steps >= max_steps:
                break
            
            # Forward
            input_ids = batch['input_ids'].to(device)
            
            step_start = time.time()
            logits, state = model(input_ids)
            
            # Loss
            targets = input_ids[:, 1:]
            logits_shifted = logits[:, :-1, :]
            loss = torch.nn.functional.cross_entropy(
                logits_shifted.reshape(-1, logits_shifted.size(-1)),
                targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step_time = time.time() - step_start
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_steps += 1
            total_steps += 1
            
            # Track edge count from graph
            if hasattr(state, 'computation_workspace'):
                graph_info = getattr(state, 'graph_info', None)
                if graph_info:
                    epoch_edges.append(graph_info.get('num_edges', 0))
            
            results['speeds'].append(step_time)
            
            if batch_idx % 50 == 0:
                ppl = torch.exp(torch.tensor(loss.item())).item()
                print(f"  Epoch {epoch+1}/{epochs}, Step {batch_idx}: "
                      f"Loss={loss.item():.4f}, PPL={ppl:.2f}, "
                      f"Speed={step_time:.3f}s")
        
        # Epoch summary (only if we actually trained)
        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            results['ppls'].append(ppl)
            
            if epoch_edges:
                avg_edges = sum(epoch_edges) / len(epoch_edges)
                results['edge_counts'].append(avg_edges)
                print(f"  Epoch {epoch+1} complete: PPL={ppl:.2f}, Avg Edges={avg_edges:.0f}")
            else:
                print(f"  Epoch {epoch+1} complete: PPL={ppl:.2f}")
        
        # Break if max_steps reached
        if total_steps >= max_steps:
            break
    
    total_time = time.time() - start_time
    
    # Final metrics
    results['final_ppl'] = results['ppls'][-1] if results['ppls'] else None
    results['avg_edges'] = sum(results['edge_counts']) / len(results['edge_counts']) if results['edge_counts'] else 0
    results['avg_speed'] = sum(results['speeds']) / len(results['speeds']) if results['speeds'] else 0
    results['total_time'] = total_time
    
    # Test homonym separation (context understanding)
    print("\nTesting homonym separation...")
    model.eval()
    with torch.no_grad():
        results['homonym_separation'] = test_homonym_separation(model, tokenizer, device)
    
    print(f"\n✓ K={k_value} Results:")
    print(f"  Final PPL: {results['final_ppl']:.2f}")
    print(f"  Avg Edges: {results['avg_edges']:.0f}")
    print(f"  Avg Speed: {results['avg_speed']:.3f}s/step")
    print(f"  Homonym Separation: {results['homonym_separation']:.4f}")
    print(f"  Total Time: {total_time/60:.1f} minutes")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='K Hyperparameter Study')
    parser.add_argument('--k-values', nargs='+', type=int, default=[3, 5, 7, 10, 12, 15],
                        help='K values to test')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per K value')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Max steps per experiment')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--output', type=str, default='k_study_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SOSM K Hyperparameter Study")
    print("="*60)
    print(f"K values: {args.k_values}")
    print(f"Epochs per K: {args.epochs}")
    print(f"Max steps: {args.max_steps}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # Run experiments
    all_results = []
    
    for k in args.k_values:
        try:
            results = train_with_k(
                k_value=k,
                epochs=args.epochs,
                device=args.device,
                max_steps=args.max_steps
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"✗ Error with K={k}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_path = PROJECT_ROOT / args.output
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': vars(args),
            'results': all_results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS & RECOMMENDATION")
    print("="*60)
    
    if not all_results:
        print("No results to analyze!")
        return
    
    # Sort by PPL
    sorted_by_ppl = sorted(all_results, key=lambda x: x['final_ppl'] or float('inf'))
    
    print("\n📊 Results Summary (sorted by PPL):")
    print(f"{'K':<5} {'PPL':<8} {'Edges':<8} {'Speed':<10} {'Homonym':<10}")
    print("-" * 50)
    for r in sorted_by_ppl:
        print(f"{r['k']:<5} {r['final_ppl']:<8.2f} {r['avg_edges']:<8.0f} "
              f"{r['avg_speed']:<10.3f} {r['homonym_separation']:<10.4f}")
    
    # Find best
    best_ppl = sorted_by_ppl[0]
    
    # Find best speed/PPL tradeoff
    tradeoffs = []
    for r in all_results:
        if r['final_ppl'] and r['avg_speed']:
            # Normalized score: lower PPL + faster speed
            score = (r['final_ppl'] / best_ppl['final_ppl']) + (r['avg_speed'] / best_ppl['avg_speed'])
            tradeoffs.append((r['k'], score, r))
    
    tradeoffs.sort(key=lambda x: x[1])
    
    print("\n🏆 Recommendations:")
    print(f"  Best PPL: K={best_ppl['k']} (PPL={best_ppl['final_ppl']:.2f})")
    
    if tradeoffs:
        best_tradeoff_k, best_tradeoff_score, best_tradeoff = tradeoffs[0]
        print(f"  Best Speed/PPL Tradeoff: K={best_tradeoff_k} "
              f"(PPL={best_tradeoff['final_ppl']:.2f}, "
              f"Speed={best_tradeoff['avg_speed']:.3f}s)")
    
    # Contextaware test
    best_homonym = sorted(all_results, key=lambda x: x.get('homonym_separation', 0), reverse=True)[0]
    if best_homonym['homonym_separation'] > 0.05:
        print(f"  ✓ Context-aware features viable! K={best_homonym['k']} "
              f"(separation={best_homonym['homonym_separation']:.4f} > 0.05)")
    else:
        print(f"  ✗ Context insufficient: Best separation={best_homonym['homonym_separation']:.4f} < 0.05")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

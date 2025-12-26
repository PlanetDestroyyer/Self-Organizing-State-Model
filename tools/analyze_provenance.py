"""
Edge Provenance Analysis Tool - Phase 2.2

Analyzes which MU blocks contribute most to semantic edge formation.

Usage:
    python tools/analyze_provenance.py --checkpoint sosm_trained.pt
"""

import sys
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from state_core.pipeline import StateCorePipeline
from transformers import GPT2Tokenizer

def analyze_provenance(model, tokenizer, num_samples=50, device='cuda'):
    """
    Analyze which MU blocks contribute to semantic edges.
    
    Returns statistics on I, R2, K block usage.
    """
    print("Analyzing edge provenance...")
    
    # Collect provenance data
    all_provenance = []
    
    # Sample texts
    texts = [
        "The bank of the river was covered with flowers.",
        "I went to the bank to deposit money.",
        "The baseball bat hit the ball.",
        "The bat flew through the dark cave.",
        "Spring is the season of flowers.",
        "The metal spring compressed easily.",
        "Python is a programming language.",
        "The python snake is very long.",
        "Apple makes smartphones and computers.",
        "An apple is a delicious fruit.",
    ] * 5  # Repeat for more samples
    
    model.eval()
    with torch.no_grad():
        for text in texts[:num_samples]:
            tokens = tokenizer.encode(text, return_tensors='pt').to(device)
            
            # Forward pass
            try:
                _, state = model(tokens)
                
                # Extract provenance if available
                if hasattr(state, 'routing_state') and state.routing_state:
                    graph = state.routing_state.get('graph', {})
                    prov = graph.get('provenance', [])
                    
                    if prov:
                        all_provenance.extend(prov)
            except Exception as e:
                print(f"Warning: Error processing text: {e}")
                continue
    
    if not all_provenance:
        print("⚠️  No provenance data found!")
        print("   Make sure track_provenance=True in graph builder")
        return None
    
    print(f"✓ Collected {len(all_provenance)} edge provenance entries")
    
    # Analyze block contributions
    block_stats = {
        'I': [],
        'R2': [],
        'K': []
    }
    
    for entry in all_provenance:
        for block in ['I', 'R2', 'K']:
            sim_key = f'{block}_similarity'
            if sim_key in entry:
                block_stats[block].append(entry[sim_key])
    
    # Compute statistics
    results = {}
    print("\n" + "="*60)
    print("BLOCK CONTRIBUTION ANALYSIS")
    print("="*60)
    
    for block, sims in block_stats.items():
        if sims:
            results[block] = {
                'mean': np.mean(sims),
                'std': np.std(sims),
                'min': np.min(sims),
                'max': np.max(sims),
                'median': np.median(sims)
            }
            
            print(f"\n{block} Block:")
            print(f"  Mean similarity: {results[block]['mean']:.4f}")
            print(f"  Std deviation:   {results[block]['std']:.4f}")
            print(f"  Min/Max:         {results[block]['min']:.4f} / {results[block]['max']:.4f}")
            print(f"  Median:          {results[block]['median']:.4f}")
    
    # Determine most important block
    mean_contributions = {b: results[b]['mean'] for b in results}
    most_important = max(mean_contributions, key=mean_contributions.get)
    
    print(f"\n🏆 Most Important Block: {most_important}")
    print(f"   (Highest mean contribution: {mean_contributions[most_important]:.4f})")
    
    # Plot distribution
    try:
        plot_provenance_distribution(block_stats, results)
        print("\n✓ Saved plot: provenance_distribution.png")
    except Exception as e:
        print(f"\n⚠️  Could not create plot: {e}")
    
    return results


def plot_provenance_distribution(block_stats, results):
    """Plot block contribution distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (block, sims) in enumerate(block_stats.items()):
        if sims:
            axes[idx].hist(sims, bins=30, alpha=0.7, edgecolor='black')
            axes[idx].axvline(results[block]['mean'], color='red', 
                            linestyle='--', label=f'Mean: {results[block]["mean"]:.3f}')
            axes[idx].set_title(f'{block} Block Similarity Distribution')
            axes[idx].set_xlabel('Cosine Similarity')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('provenance_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze edge provenance')
    parser.add_argument('--checkpoint', type=str, default='sosm_trained.pt',
                       help='Model checkpoint to analyze')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of text samples to analyze')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EDGE PROVENANCE ANALYSIS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    
    # Create config (minimal, will be overridden by checkpoint)
    config = {
        'stage': 3,
        'components': {
            'mu': {'vocab_size': 50257, 'embed_dim': 64},
            'temporal': {},
            'graph': {'semantic_k': 10}
        },
        'model': {'hidden_dim': 896, 'n_layers': 4, 'n_heads': 8}
    }
    
    model = StateCorePipeline(config).to(args.device)
    
    # Load checkpoint
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"⚠️  Checkpoint not found: {args.checkpoint}")
        print("   Analyzing untrained model (for testing)")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Analyze
    results = analyze_provenance(model, tokenizer, args.num_samples, args.device)
    
    if results:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nInterpretation:")
        print("  - High I contrib: Identity/semantics drive connections")
        print("  - High R2 contrib: Relations/context drive connections")
        print("  - High K contrib: Knowledge/concepts drive connections")
        print("\nUse this to prune unused blocks or adjust semantic_blocks config!")
    
    return results


if __name__ == '__main__':
    main()

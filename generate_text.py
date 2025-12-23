#!/usr/bin/env python3
"""
Text Generation Comparison: SOSM vs Baseline

Loads both trained models and generates text from prompts.

Usage:
    python generate_text.py
    python generate_text.py --prompt "The quick brown"
    python generate_text.py --max-tokens 50
"""

import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from state_core import StateCorePipeline
from test_base import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_sosm(checkpoint_path: str = 'sosm_stage3_checkpoint.pt'):
    """Load SOSM model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = {
        'stage': checkpoint.get('stage', 3),
        'components': {
            'mu': {'vocab_size': checkpoint['vocab_size'], 'embed_dim': 64, 'max_seq_len': 64},
            'temporal': {'time_dim': 32},
            'k1': {},
            'graph': {'sequential_edges': True, 'semantic_edges': True}
        },
        'model': {'hidden_dim': 256, 'n_layers': 4, 'n_heads': 4, 'dropout': 0.1}
    }
    
    model = StateCorePipeline(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['vocab_size']


def load_baseline(checkpoint_path: str = 'baseline_checkpoint.pt'):
    """Load baseline model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = BaselineTransformer(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=256, n_layers=4, n_heads=4, ff_dim=512, max_seq_len=64, dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['vocab_size']


def generate(model, prompt_ids: list, max_tokens: int = 30, temperature: float = 0.8,
             top_k: int = 40, vocab_size: int = 10000, is_sosm: bool = False,
             repetition_penalty: float = 1.2):
    """Generate text using temperature sampling with repetition penalty."""
    input_ids = prompt_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([input_ids[-64:]], dtype=torch.long, device=device)
            
            if is_sosm:
                logits, _ = model(x)
            else:
                logits = model(x)
            
            next_logits = logits[0, -1, :] / temperature
            
            # Apply repetition penalty
            for prev_token in set(input_ids[-20:]):
                if prev_token < len(next_logits):
                    next_logits[prev_token] /= repetition_penalty
            
            # Top-k sampling
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, len(next_logits)))
                next_logits[next_logits < values[-1]] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            input_ids.append(next_token)
    
    return input_ids


def build_vocab_from_data():
    """Build vocabulary using same fixed charset as sosm_data.py."""
    # Use same comprehensive character set as sosm_data.py
    chars = (
        " \t\n"  # Whitespace
        "abcdefghijklmnopqrstuvwxyz"  # Lowercase
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Uppercase
        "0123456789"  # Digits
        ".,;:!?'\"-()[]{}"  # Punctuation
        "@#$%^&*+=<>/\\|`~_"  # Symbols (for code)
        "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"  # Extended Latin
    )
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    print(f"  Using fixed vocab: {len(chars)} chars")
    return char_to_idx, idx_to_char


def load_vocab_from_checkpoint(checkpoint_path: str):
    """Try to load vocabulary from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'char_to_idx' in checkpoint and 'idx_to_char' in checkpoint:
            print(f"  ✓ Loaded vocab from checkpoint")
            return checkpoint['char_to_idx'], checkpoint['idx_to_char']
    except:
        pass
    return None, None


def tokens_to_text(token_ids: list, idx_to_char: dict) -> str:
    """Convert token IDs back to text."""
    return ''.join(idx_to_char.get(t, '?') for t in token_ids)


def main():
    parser = argparse.ArgumentParser(description='Text Generation Comparison')
    parser.add_argument('--prompt', type=str, default='The ', help='Starting prompt')
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    args = parser.parse_args()
    
    print("=" * 70)
    print("TEXT GENERATION COMPARISON: SOSM vs BASELINE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    print()
    
    # Build vocabulary from training data
    print("Building vocabulary from training data...")
    char_to_idx, idx_to_char = build_vocab_from_data()
    
    # Tokenize prompt
    prompt_ids = [char_to_idx.get(c, 0) for c in args.prompt]
    print(f"Prompt tokens: {prompt_ids[:10]}...")
    print()
    
    # Load and generate with SOSM
    print("Loading SOSM model...")
    try:
        sosm, vocab_size = load_sosm()
        print(f"  ✓ Loaded (vocab={vocab_size})")
        
        print("\nGenerating with SOSM...")
        sosm_ids = generate(sosm, prompt_ids, args.max_tokens, args.temperature,
                           top_k=args.top_k, vocab_size=vocab_size, is_sosm=True,
                           repetition_penalty=1.3)
        sosm_text = tokens_to_text(sosm_ids, idx_to_char)
        print(f"\n📗 SOSM Output:\n{sosm_text}")
    except FileNotFoundError:
        print("  ✗ sosm_stage3_checkpoint.pt not found")
        sosm_text = None
    
    print("\n" + "-" * 70)
    
    # Load and generate with Baseline
    print("\nLoading Baseline model...")
    try:
        baseline, vocab_size = load_baseline()
        print(f"  ✓ Loaded (vocab={vocab_size})")
        
        print("\nGenerating with Baseline...")
        base_ids = generate(baseline, prompt_ids, args.max_tokens, args.temperature,
                           top_k=args.top_k, vocab_size=vocab_size, is_sosm=False,
                           repetition_penalty=1.3)
        base_text = tokens_to_text(base_ids, idx_to_char)
        print(f"\n📘 Baseline Output:\n{base_text}")
    except FileNotFoundError:
        print("  ✗ baseline_checkpoint.pt not found")
        base_text = None
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

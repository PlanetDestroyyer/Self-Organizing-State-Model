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
    
    vocab_size = checkpoint.get('vocab_size', 50257)
    
    config = {
        'stage': checkpoint.get('stage', 3),
        'components': {
            'mu': {'vocab_size': vocab_size, 'embed_dim': 64, 'max_seq_len': 64},
            'temporal': {'time_dim': 32},
            'k1': {},
            'graph': {'sequential_edges': True, 'semantic_edges': True}
        },
        'model': {'hidden_dim': 256, 'n_layers': 4, 'n_heads': 4, 'dropout': 0.1}
    }
    
    model = StateCorePipeline(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, vocab_size, checkpoint


def load_baseline(checkpoint_path: str = 'baseline_checkpoint.pt'):
    """Load baseline model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint.get('vocab_size', 50257)
    
    model = BaselineTransformer(
        vocab_size=vocab_size,
        embed_dim=256, n_layers=4, n_heads=4, ff_dim=512, max_seq_len=64, dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, vocab_size, checkpoint


def generate(model, prompt_ids: list, max_tokens: int = 30, temperature: float = 0.8,
             top_k: int = 40, repetition_penalty: float = 1.2):
    """Generate text using temperature sampling with repetition penalty."""
    input_ids = prompt_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Context window
            ctx = input_ids[-64:]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            
            # Forward
            if isinstance(model, StateCorePipeline):
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


class TokenizerWrapper:
    """Handles both BPE and Char-level tokenization."""
    def __init__(self, checkpoint=None):
        self.type = 'char'
        self.tokenizer = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Try to load from checkpoint
        if checkpoint:
            if 'char_to_idx' in checkpoint and isinstance(checkpoint['char_to_idx'], dict):
                # Legacy or Char (check if it's actually BPE info stored in dict)
                vocab_info = checkpoint['char_to_idx']
                if vocab_info.get('type') == 'bpe':
                    self.init_bpe()
                else:
                    print("  Using Char tokenizer from checkpoint")
                    self.char_to_idx = checkpoint['char_to_idx']
                    self.idx_to_char = checkpoint['idx_to_char']
            else:
                # Default to BPE if no char vocab found (new default)
                self.init_bpe()
        else:
            self.init_bpe()
            
    def init_bpe(self):
        try:
            from transformers import GPT2TokenizerFast
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            self.type = 'bpe'
            print("  Using GPT-2 BPE tokenizer")
        except:
            print("  Warning: GPT-2 tokenizer not found, using char fallback")
            self.type = 'char'
            chars = " \t\n" + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789" + ".,;:!?'\"-()[]{}"
            self.char_to_idx = {c: i for i, c in enumerate(chars)}
            self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

    def encode(self, text):
        if self.type == 'bpe':
            return self.tokenizer.encode(text)
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, token_ids):
        if self.type == 'bpe':
            return self.tokenizer.decode(token_ids)
        return ''.join(self.idx_to_char.get(t, '?') for t in token_ids)


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
    
    # Load SOSM
    print("Loading SOSM model...")
    try:
        sosm, vocab_size, sosm_ckpt = load_sosm()
        print(f"  ✓ Loaded (vocab={vocab_size})")
    except FileNotFoundError:
        print("  ✗ sosm_stage3_checkpoint.pt not found")
        sosm = None
        sosm_ckpt = None

    # Load Baseline
    print("\nLoading Baseline model...")
    try:
        baseline, base_vocab, base_ckpt = load_baseline()
        print(f"  ✓ Loaded (vocab={base_vocab})")
    except FileNotFoundError:
        print("  ✗ baseline_checkpoint.pt not found")
        baseline = None
        base_ckpt = None
        
    # Use tokenizer from SOSM checkpoint if available, else Baseline, else default
    print("\nInitializing Tokenizer...")
    ckpt_to_use = sosm_ckpt if sosm_ckpt else base_ckpt
    tokenizer = TokenizerWrapper(ckpt_to_use)
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Prompt tokens: {prompt_ids}")
    print()
    
    # Generate SOSM
    if sosm:
        print("Generating with SOSM...")
        sosm_ids = generate(sosm, prompt_ids, args.max_tokens, args.temperature,
                           top_k=args.top_k, repetition_penalty=1.3)
        sosm_text = tokenizer.decode(sosm_ids)
        print(f"\n📗 SOSM Output:\n{sosm_text}")
    
    print("\n" + "-" * 70)
    
    # Generate Baseline
    if baseline:
        print("\nGenerating with Baseline...")
        base_ids = generate(baseline, prompt_ids, args.max_tokens, args.temperature,
                           top_k=args.top_k, repetition_penalty=1.3)
        base_text = tokenizer.decode(base_ids)
        print(f"\n📘 Baseline Output:\n{base_text}")
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

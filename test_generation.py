#!/usr/bin/env python3
"""
Quick Generation Test with Different Parameters

Tests SOSM generation with various temperature, top_k, and repetition_penalty settings.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from state_core import StateCorePipeline
from test_base import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_sosm():
    checkpoint = torch.load('sosm_stage3_checkpoint.pt', map_location=device)
    vocab_size = checkpoint.get('vocab_size', 50257)
    
    config = {
        'stage': 3,
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
    return model, vocab_size


def generate(model, prompt_ids, max_tokens=30, temperature=0.7, 
             top_k=30, top_p=0.9, repetition_penalty=2.5):
    """Generate with tuned parameters for anti-repetition."""
    input_ids = prompt_ids.copy()
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_tokens):
            ctx = input_ids[-64:]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            
            if isinstance(model, StateCorePipeline):
                logits, _ = model(x)
            else:
                logits = model(x)
            
            next_logits = logits[0, -1, :].clone()
            
            # Apply temperature
            next_logits = next_logits / temperature
            
            # Strong repetition penalty on recent tokens
            recent = input_ids[-30:]  # Look at last 30 tokens
            for i, token in enumerate(recent):
                if token < len(next_logits):
                    # Stronger penalty for more recent tokens
                    recency_factor = 1 + (i / len(recent))  # 1.0 to 2.0
                    next_logits[token] /= repetition_penalty * recency_factor
            
            # Penalty for generated tokens (even stronger)
            for token in set(generated_tokens):
                if token < len(next_logits):
                    next_logits[token] /= repetition_penalty * 2
            
            # Top-k filtering
            if top_k > 0:
                k = min(top_k, len(next_logits))
                values, indices = torch.topk(next_logits, k)
                mask = torch.ones_like(next_logits) * float('-inf')
                mask[indices] = next_logits[indices]
                next_logits = mask
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[1:] = sorted_mask[:-1].clone()
                sorted_mask[0] = False
                indices_to_remove = sorted_indices[sorted_mask]
                next_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            input_ids.append(next_token)
            generated_tokens.append(next_token)
    
    return input_ids


def main():
    print("=" * 70)
    print("SOSM GENERATION TEST - Tuned Parameters")
    print("=" * 70)
    
    # Load tokenizer
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        print("✓ GPT-2 tokenizer loaded")
    except:
        print("✗ Could not load tokenizer")
        return
    
    # Load model
    print("\nLoading SOSM...")
    try:
        model, vocab_size = load_sosm()
        print(f"✓ Loaded (vocab={vocab_size})")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Test prompts
    prompts = [
        "The capital of India is",
        "The quick brown fox",
        "import torch",
        "def factorial(n):",
        "Once upon a time",
        "The future of AI"
    ]
    
    # Parameter sets to test
    param_sets = [
        {"temperature": 0.5, "top_k": 20, "top_p": 0.85, "repetition_penalty": 2.5, "name": "Conservative"},
        {"temperature": 0.7, "top_k": 30, "top_p": 0.9, "repetition_penalty": 3.0, "name": "Balanced"},
        {"temperature": 0.3, "top_k": 10, "top_p": 0.8, "repetition_penalty": 4.0, "name": "Greedy-ish"},
    ]
    
    print("\n" + "=" * 70)
    
    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        print("-" * 60)
        
        prompt_ids = tokenizer.encode(prompt)
        
        for params in param_sets:
            result_ids = generate(
                model, prompt_ids, 
                max_tokens=40,
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                repetition_penalty=params["repetition_penalty"]
            )
            
            result_text = tokenizer.decode(result_ids)
            print(f"\n  [{params['name']}] (T={params['temperature']}, k={params['top_k']}, pen={params['repetition_penalty']})")
            print(f"  → {result_text}")
    
    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

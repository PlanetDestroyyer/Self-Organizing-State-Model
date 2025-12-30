"""
Generation Tests for SOSM vs Baseline

Tests both models on various generation tasks:
- Natural language (Wikipedia-style)
- Code (Python)
- Scientific text (ArXiv-style)
- Dialogue
- Creative writing
"""
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from pathlib import Path
import json


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    is_sosm: bool = True
) -> str:
    """Generate text from prompt using nucleus sampling"""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            if is_sosm:
                logits, _ = model(input_ids, return_state=True)
            else:
                logits = model(input_ids)
            
            # Get logits for last token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Stop if too long
            if input_ids.size(1) >= 512:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def run_generation_tests(sosm_model, baseline_model, tokenizer, save_dir='results'):
    """Run comprehensive generation tests"""
    
    print("\n" + "="*70)
    print("GENERATION TESTS")
    print("="*70)
    
    # Test prompts for different domains
    test_prompts = {
        'natural_language': [
            "The capital of France is",
            "In machine learning, a neural network is",
            "The history of computers began in"
        ],
        'code': [
            "def fibonacci(n):\n    ",
            "# Function to sort a list\ndef sort_list(arr):\n    ",
            "import numpy as np\n\n# Create array\narr = "
        ],
        'scientific': [
            "The fundamental principle of quantum mechanics states that",
            "In the study of neural networks, backpropagation is",
            "The theory of relativity describes"
        ],
        'creative': [
            "Once upon a time, in a distant galaxy",
            "The detective entered the room and noticed",
            "My favorite memory from childhood is"
        ]
    }
    
    results = {}
    
    for domain, prompts in test_prompts.items():
        print(f"\n{'='*70}")
        print(f"Domain: {domain.upper()}")
        print(f"{'='*70}")
        
        results[domain] = {'sosm': [], 'baseline': []}
        
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}: \"{prompt}\"")
            
            # SOSM generation
            sosm_output = generate_text(sosm_model, tokenizer, prompt, is_sosm=True)
            print(f"\n[SOSM]:\n{sosm_output}")
            results[domain]['sosm'].append({
                'prompt': prompt,
                'generation': sosm_output
            })
            
            # Baseline generation
            baseline_output = generate_text(baseline_model, tokenizer, prompt, is_sosm=False)
            print(f"\n[BASELINE]:\n{baseline_output}")
            results[domain]['baseline'].append({
                'prompt': prompt,
                'generation': baseline_output
            })
            
            print("-" * 70)
    
    # Save results
    save_path = Path(save_dir) / 'generation_tests.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Generation tests saved to: {save_path}")
    
    return results


if __name__ == "__main__":
    print("This module is imported by run_comparison.py")
    print("Run: python run_comparison.py to execute full comparison")

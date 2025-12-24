#!/usr/bin/env python3
"""
Test improved generation with repetition penalty on trained model
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from state_core import StateCorePipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint
print("Loading checkpoint...")
checkpoint = torch.load('sosm_stage0_FIXED.pt', map_location=device)

# Create model
config = checkpoint['config']
model = StateCorePipeline(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

print(f"\n{'='*70}")
print(f"IMPROVED GENERATION TEST")
print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print(f"Stage: {config['stage']}")
print(f"Checkpoint PPL: {checkpoint.get('val_loss', 0):.2f}")
print(f"{'='*70}\n")

prompts = [
    "The capital of India is",
    "Once upon a time",
    "The quick brown fox",
    "Albert Einstein was",
    "In the year 2025"
]

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = []

    with torch.no_grad():
        for _ in range(30):  # Generate 30 tokens
            logits, _ = model(input_ids)
            next_logits = logits[0, -1, :].clone()

            # Repetition penalty (penalize already used tokens)
            for token_id in set(input_ids[0].tolist() + generated):
                if token_id < len(next_logits):
                    next_logits[token_id] /= 1.8  # Strong penalty

            # Temperature (add diversity)
            next_logits = next_logits / 0.7  # Lower temp = more focused

            # Top-k filtering (only consider top 50 tokens)
            top_k = 50
            values, indices = torch.topk(next_logits, min(top_k, len(next_logits)))
            filtered_logits = torch.full_like(next_logits, float('-inf'))
            filtered_logits[indices] = values

            # Sample from filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            generated.append(next_token.item())

            input_ids = torch.cat([input_ids, next_token], dim=1)

    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Prompt: '{prompt}'")
    print(f"Output: {output}")
    print()

print(f"{'='*70}")
print("\nNote: If still repetitive, the model needs more training epochs.")
print("Your 155 PPL is excellent! Try Stage 1 (TEMPORAL) next.")
print(f"{'='*70}\n")

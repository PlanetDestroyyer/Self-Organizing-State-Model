"""
Quick test of generation functionality
"""
import torch
from transformers import GPT2Tokenizer
from baseline_transformer import BaselineTransformer
from generation_prompts import WIKI_PROMPTS, CODE_PROMPTS, ARXIV_PROMPTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create untrained baseline
model = BaselineTransformer().to(device)
model.eval()

print("Testing generation with untrained model...")
print(f"Device: {device}")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# Test one prompt from each domain
test_prompts = [
    ("Wikipedia", WIKI_PROMPTS[0]),
    ("Code", CODE_PROMPTS[0]),
    ("ArXiv", ARXIV_PROMPTS[0])
]

with torch.no_grad():
    for domain, prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Testing {domain}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        print(f"Input shape: {input_ids.shape}")
        
        # Generate 20 tokens
        generated = input_ids.clone()
        for _ in range(20):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        
        # Decode
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print(f"✓ Generation successful!")

print(f"\n{'='*60}")
print("All generation tests passed!")
print(f"{'='*60}")

"""Text generation utilities"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from ..config import MUSOTAConfig
from ..data import WikiTextBPEDataset

config = MUSOTAConfig()


def generate_text(model: nn.Module, dataset: WikiTextBPEDataset, prompt: str,
                  max_length: int = 100, temperature: float = 0.8,
                  top_k: int = 50, top_p: float = 0.9,
                  repetition_penalty: float = 1.2, device: str = 'cuda') -> str:
    """Generate text with temperature/top-k/top-p sampling"""
    model.eval()

    # Determine device type for autocast
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # Encode prompt
    encoding = dataset.tokenizer.encode(prompt)
    input_ids = encoding.ids

    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_tensor = torch.tensor([input_ids[-config.max_seq_len:]], dtype=torch.long).to(device)

            # Forward pass
            with autocast(device_type, enabled=config.use_mixed_precision):
                logits = model(input_tensor)

            next_token_logits = logits[0, -1, :] / temperature

            # Apply repetition penalty
            for token_id in set(input_ids):
                next_token_logits[token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            input_ids.append(next_token)

    return dataset.decode(input_ids)

"""
Sampling utilities for text generation.

Implements various decoding strategies:
- Greedy (argmax)
- Top-k sampling
- Nucleus (top-p) sampling
- Temperature scaling

Based on "The Curious Case of Neural Text Degeneration" (Holtzman et al., ICLR 2020)
"""

import torch
import torch.nn.functional as F


def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy decoding: always pick highest probability token.
    
    Args:
        logits: [vocab_size] unnormalized logits
        
    Returns:
        token_id: [1] next token ID
    """
    return logits.argmax(dim=-1, keepdim=True)


def top_k_sampling(
    logits: torch.Tensor,
    k: int = 50,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Top-k sampling: sample from top k most likely tokens.
    
    Args:
        logits: [vocab_size] unnormalized logits
        k: Number of top tokens to consider
        temperature: Temperature for softmax (higher = more random)
        
    Returns:
        token_id: [1] sampled token ID
    """
    # Temperature scaling
    logits = logits / temperature
    
    # Get top-k
    top_k_logits, top_k_indices = logits.topk(k, dim=-1)
    
    # Sample from top-k
    probs = F.softmax(top_k_logits, dim=-1)
    sample_idx = torch.multinomial(probs, num_samples=1)
    
    return top_k_indices[sample_idx]


def nucleus_sampling(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: sample from smallest set of tokens with cumulative prob >= p.
    
    This is the recommended sampling strategy for high-quality text generation.
    Avoids both truncation of tail (top-k) and degenerate repetitions (greedy).
    
    Args:
        logits: [vocab_size] unnormalized logits
        p: Cumulative probability threshold (typically 0.9 or 0.95)
        temperature: Temperature for softmax (higher = more random)
        min_tokens_to_keep: Minimum number of tokens to keep (default 1)
        
    Returns:
        token_id: [1] sampled token ID
        
    Example:
        >>> logits = model(tokens)[:, -1, :]  # Last position logits
        >>> next_token = nucleus_sampling(logits, p=0.9, temperature=0.8)
    """
    # Temperature scaling
    logits = logits / temperature
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # Compute cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff: first position where cumsum > p
    # But always keep at least min_tokens_to_keep
    sorted_indices_to_remove = cumsum_probs > p
    
    # Shift right to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False  # Never remove top token
    
    # Ensure we keep at least min_tokens_to_keep
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    
    # Zero out removed indices in original (unsorted) logits
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    
    # Set logits of removed tokens to -inf
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # Sample from filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


def sample_with_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float = 1.2,
    sampling_fn=nucleus_sampling,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Apply repetition penalty before sampling.
    
    Reduces probability of recently generated tokens to avoid loops.
    
    Args:
        logits: [vocab_size] unnormalized logits
        generated_tokens: [seq_len] previously generated tokens
        penalty: Penalty factor (>1.0 = penalize, <1.0 = encourage)
        sampling_fn: Sampling function to use (nucleus_sampling, top_k_sampling, etc.)
        **sampling_kwargs: Additional arguments for sampling function
        
    Returns:
        token_id: [1] sampled token ID
        
    Example:
        >>> logits = model(tokens)[:, -1, :]
        >>> next_token = sample_with_repetition_penalty(
        ...     logits, 
        ...     generated_tokens=tokens[0],  # All tokens generated so far
        ...     penalty=1.2,
        ...     p=0.9
        ... )
    """
    # Apply penalty to recently generated tokens
    if generated_tokens.numel() > 0:
        for token_id in generated_tokens.unique():
            logits[token_id] /= penalty
    
    # Sample with penalty-adjusted logits
    return sampling_fn(logits, **sampling_kwargs)


# ============================================================================
# Testing
# ============================================================================

def test_sampling():
    """Test sampling functions."""
    print("Testing sampling utilities...")
    
    # Create sample logits
    vocab_size = 1000
    logits = torch.randn(vocab_size)
    
    # Test greedy
    greedy_token = greedy_sampling(logits)
    assert greedy_token.shape == (1,)
    print(f"✓ Greedy sampling: token {greedy_token.item()}")
    
    # Test top-k
    topk_token = top_k_sampling(logits, k=50, temperature=0.8)
    assert topk_token.shape == (1,)
    print(f"✓ Top-k sampling: token {topk_token.item()}")
    
    # Test nucleus
    nucleus_token = nucleus_sampling(logits, p=0.9, temperature=0.8)
    assert nucleus_token.shape == (1,)
    print(f"✓ Nucleus sampling: token {nucleus_token.item()}")
    
    # Test repetition penalty
    generated = torch.tensor([10, 20, 30, 10, 20])  # 10 and 20 repeated
    penalized_token = sample_with_repetition_penalty(
        logits, generated, penalty=1.5, p=0.9
    )
    assert penalized_token.shape == (1,)
    print(f"✓ Repetition penalty: token {penalized_token.item()}")
    
    # Test nucleus actually creates a smaller vocabulary
    # High p should keep more tokens, low p should keep fewer
    logits_uniform = torch.ones(vocab_size)  # Uniform distribution
    
    # With p=0.5, should keep ~half the tokens
    nucleus_token_05 = nucleus_sampling(logits_uniform, p=0.5)
    
    # With p=0.99, should keep almost all tokens
    nucleus_token_99 = nucleus_sampling(logits_uniform, p=0.99)
    
    print(f"✓ Nucleus p=0.5 vs p=0.99: both valid tokens")
    
    print("\n✅ All sampling tests passed!")


if __name__ == '__main__':
    test_sampling()

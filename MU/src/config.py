"""Configuration for MU-SOTA Transformer"""

import torch


class MUSOTAConfig:
    """Configuration for MU-SOTA Transformer"""

    # Matrix structure (8×8 = 64 dims, 16 semantic blocks)
    matrix_size = 8
    num_semantic_blocks = 16  # Each block is 2×2
    block_size = 2  # 2×2 blocks

    # Architecture (SOTA-level)
    n_layers = 12  # Scaled up from 6 for better performance
    n_heads = 8
    dropout = 0.1

    # Vocabulary
    vocab_size = 50000  # Like GPT-2
    max_seq_len = 256  # Increased from 128 for better context (memory-safe)

    # Training
    batch_size = 12  # Reduced to compensate for longer sequences
    num_epochs = 100  # Increased from 10 to break through plateau
    learning_rate = 1e-4  # REDUCED from 3e-4 to help with convergence
    weight_decay = 0.01
    warmup_steps = 2000  # INCREASED from 500 for smoother start
    max_grad_norm = 1.0
    gradient_accumulation_steps = 8  # Effective batch_size = 32 (maintained)

    # Mixed precision
    use_mixed_precision = True

    # Generation
    temperature = 0.8
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.2

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

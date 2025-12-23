"""
Model configuration classes for MU Transformer and Baseline Transformer
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MUConfig:
    """Configuration for MU Transformer model"""

    # Vocabulary
    vocab_size: int = 30000

    # MU matrix dimensions
    r: int = 4  # Number of rows in MU matrix
    c: int = 4  # Number of columns in MU matrix

    # Model architecture
    d_model: int = 128  # Hidden dimension for attention
    n_layers: int = 6  # Number of transformer layers
    n_heads: int = 4  # Number of attention heads

    # Regularization
    dropout: float = 0.1

    # Sequence
    max_seq_len: int = 256

    # Sensitivity mask for different slots
    # Controls how much each position in MU matrix can change
    sensitivity_mask: Optional[list] = None

    def __post_init__(self):
        """Initialize sensitivity mask if not provided"""
        if self.sensitivity_mask is None:
            # Default sensitivity mask based on semantic roles
            # Shape: [r, c] = [4, 4]
            self.sensitivity_mask = [
                [0.1, 0.01, 0.01, 0.7],   # Row 0: Identity (low), Invariants (very low), Relation (high)
                [0.7, 0.7, 0.7, 0.9],      # Row 1: Relations (high), Context start (very high)
                [0.9, 0.9, 0.9, 0.6],      # Row 2: Context (very high), Transform (medium-high)
                [0.6, 0.5, 0.5, 0.1]       # Row 3: Transform (medium-high), Compositional (medium), Global (low)
            ]

        # Validate dimensions
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.r * self.c <= self.d_model, \
            f"MU size ({self.r * self.c}) should not exceed d_model ({self.d_model})"


@dataclass
class BaselineConfig:
    """Configuration for baseline transformer model (for comparison)"""

    # Vocabulary
    vocab_size: int = 30000

    # Model architecture
    d_model: int = 128  # Hidden dimension (embedding size)
    n_layers: int = 6  # Number of transformer layers
    n_heads: int = 4  # Number of attention heads
    d_ff: int = 512  # Feedforward dimension

    # Regularization
    dropout: float = 0.1

    # Sequence
    max_seq_len: int = 256

    def __post_init__(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


def get_config_from_dict(config_dict: dict, model_type: str = 'mu'):
    """
    Create a config object from a dictionary

    Args:
        config_dict: Dictionary containing configuration parameters
        model_type: 'mu' or 'baseline'

    Returns:
        MUConfig or BaselineConfig object
    """
    if model_type == 'mu':
        return MUConfig(**config_dict)
    elif model_type == 'baseline':
        return BaselineConfig(**config_dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

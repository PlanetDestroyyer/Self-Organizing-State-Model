"""
TEMPORAL Module

Self-learning time embeddings for temporal pattern discovery.
"""

import sys
from pathlib import Path

# Add TEMPORAL to path
_proto_path = Path(__file__).parent / "TEMPORAL"
if str(_proto_path) not in sys.path:
    sys.path.insert(0, str(_proto_path))

# Core exports (all prefixed with Temporal_)
try:
    from model import (
        Temporal_Transformer,
        Temporal_Baseline,
        Temporal_RMSNorm,
        Temporal_SwiGLU,
        Temporal_Attention,
        Temporal_Block
    )

    from time_embeddings import (
        Temporal_TimeEmbeddings,
        Temporal_Tokenizer
    )

    from config import (
        Temporal_Config,
        Temporal_ColabConfig,
        Temporal_ScaledConfig,
        Temporal_DebugConfig,
        get_config
    )
    
except ImportError as e:
    print(f"Warning: TEMPORAL imports failed: {e}")
    Temporal_Transformer = None
    Temporal_Baseline = None
    Temporal_TimeEmbeddings = None
    Temporal_Tokenizer = None
    Temporal_Config = None

__all__ = [
    'Temporal_Transformer',
    'Temporal_Baseline',
    'Temporal_TimeEmbeddings',
    'Temporal_Tokenizer',
    'Temporal_Config',
    'Temporal_ColabConfig',
    'Temporal_ScaledConfig',
    'Temporal_DebugConfig',
    'Temporal_RMSNorm',
    'Temporal_SwiGLU',
    'Temporal_Attention',
    'Temporal_Block',
    'get_config',
]

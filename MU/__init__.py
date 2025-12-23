"""
MU (Meaning Unit) Module

8×8 structured semantic matrices for meaning representation.
"""

import sys
from pathlib import Path

# Add src to path for internal imports
_src_path = Path(__file__).parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Core exports (all prefixed with MU_)
from .mu_sota import (
    MU_Config,
    MU_Transformer,
    MU_SemanticBlockLayout,
    MU_BlockAttention,
    MU_DynamicSensitivity,
    MU_Dataset
)

__all__ = [
    'MU_Config',
    'MU_Transformer',
    'MU_SemanticBlockLayout',
    'MU_BlockAttention',
    'MU_DynamicSensitivity',
    'MU_Dataset',
]

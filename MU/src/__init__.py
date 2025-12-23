"""
MU-SOTA: Meaning Unit Transformer with State-of-the-Art Architecture

A modular implementation of the MU Transformer with:
- 8x8 structured semantic matrix (16 blocks of 2x2)
- Block-wise semantic attention (structure-aware)
- 24-layer deep architecture (SOTA depth)
- Mixed precision training (FP16 + FP32)
- Temperature sampling generation
- 50K BPE vocabulary
"""

__version__ = '1.0.0'

"""
State Core Integration Layer

Wraps MU, TEMPORAL, and K-1 repos as black boxes.
"""

from .state import State
from .pipeline import StateCorePipeline
from .stages import StageController

__all__ = ['State', 'StateCorePipeline', 'StageController']

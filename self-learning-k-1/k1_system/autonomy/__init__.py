"""
Autonomy components for Phase 2 self-learning.

This module provides:
- BoundarySystem: Staged autonomy with intelligent boundary testing
- Phase2Controller: Manages Phase 1 → Phase 2 transition
- Action: Represents autonomous actions
- Stages: Stage definitions and configurations
"""

from .parameter_controller import ParameterController, SystemState
from .stopping_controller import StoppingController
from .self_diagnostic import SelfDiagnostic
from .stages import AutonomyStage, BoundaryStage, STAGE_CONFIGS
from .actions import Action, ACTION_ADD_AGENT, ACTION_DELETE_AGENT, ACTION_TUNE_PARAMETER, ACTION_STOP_TRAINING
from .boundary_system import BoundarySystem, Phase2Controller

__all__ = [
    # Existing
    'ParameterController',
    'SystemState',
    'StoppingController',
    'SelfDiagnostic',
    # New - Stages
    'AutonomyStage',
    'BoundaryStage',
    'STAGE_CONFIGS',
    # New - Actions
    'Action',
    'ACTION_ADD_AGENT',
    'ACTION_DELETE_AGENT', 
    'ACTION_TUNE_PARAMETER',
    'ACTION_STOP_TRAINING',
    # New - Boundary System
    'BoundarySystem',
    'Phase2Controller',
]

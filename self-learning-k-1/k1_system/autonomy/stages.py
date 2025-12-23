"""
Autonomy Stage definitions for Phase 2.

Each stage defines what actions are allowed and the criteria
for advancing to the next stage.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict


class AutonomyStage(Enum):
    """The 4 stages of increasing autonomy."""
    STAGE_1_ADD_ONLY = 1
    STAGE_2_PARAMETER = 2
    STAGE_3_STRUCTURAL = 3
    STAGE_4_FULL = 4


@dataclass
class BoundaryStage:
    """
    Configuration for a single autonomy stage.
    
    Attributes:
        stage: The autonomy stage enum value
        can_add: Whether adding agents is allowed
        can_delete: Whether deleting agents is allowed
        can_tune_params: Whether parameter tuning is allowed
        can_stop: Whether self-stopping is allowed
        param_bounds: Dict of parameter name to (min, max) bounds
        min_agents: Minimum number of agents (safety constraint)
        cheats_to_advance: Number of successful cheats needed to advance
    """
    stage: AutonomyStage
    can_add: bool = False
    can_delete: bool = False
    can_tune_params: bool = False
    can_stop: bool = False
    param_bounds: Dict[str, tuple] = field(default_factory=dict)
    min_agents: int = 10
    cheats_to_advance: int = 3


# Predefined stage configurations
STAGE_CONFIGS = {
    1: BoundaryStage(
        stage=AutonomyStage.STAGE_1_ADD_ONLY,
        can_add=True,
        can_delete=False,
        can_tune_params=False,
        can_stop=False,
        cheats_to_advance=3
    ),
    2: BoundaryStage(
        stage=AutonomyStage.STAGE_2_PARAMETER,
        can_add=True,
        can_delete=False,
        can_tune_params=True,
        can_stop=False,
        param_bounds={
            'learning_rate': (0.0001, 0.01),
            'cooldown_steps': (5, 50),
            'top_k': (3, 10),
            'batch_size': (16, 512)
        },
        cheats_to_advance=5
    ),
    3: BoundaryStage(
        stage=AutonomyStage.STAGE_3_STRUCTURAL,
        can_add=True,
        can_delete=True,
        can_tune_params=True,
        can_stop=True,
        param_bounds={
            'learning_rate': (0.00001, 0.1),
            'cooldown_steps': (1, 100),
            'top_k': (1, 20),
            'batch_size': (8, 1024)
        },
        min_agents=10,
        cheats_to_advance=10
    ),
    4: BoundaryStage(
        stage=AutonomyStage.STAGE_4_FULL,
        can_add=True,
        can_delete=True,
        can_tune_params=True,
        can_stop=True,
        min_agents=5,
        cheats_to_advance=999999  # Never advance past this
    )
}

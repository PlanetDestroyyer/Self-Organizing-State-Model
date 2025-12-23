"""
Stage Controller for the Self-Organizing State Model.

Defines what is active at each stage - no hard-coded flags.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class StageConfig:
    """Configuration for a single stage."""
    mu: bool = True
    temporal: bool = False
    k1: bool = False
    graph: bool = False


class StageController:
    """
    Controls which components are active at each stage.
    
    Stages:
        0: MU only (baseline)
        1: MU + TEMPORAL
        2: MU + TEMPORAL + K-1
        3: MU + TEMPORAL + K-1 + Graph Routing
    """
    
    STAGES: Dict[int, StageConfig] = {
        0: StageConfig(mu=True, temporal=False, k1=False, graph=False),
        1: StageConfig(mu=True, temporal=True,  k1=False, graph=False),
        2: StageConfig(mu=True, temporal=True,  k1=True,  graph=False),
        3: StageConfig(mu=True, temporal=True,  k1=True,  graph=True),
    }
    
    def __init__(self, stage: int = 0):
        """
        Initialize stage controller.
        
        Args:
            stage: Stage number (0-3)
        """
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage {stage}. Must be 0-3.")
        
        self._stage = stage
        self._config = self.STAGES[stage]
    
    @property
    def stage(self) -> int:
        return self._stage
    
    @property
    def config(self) -> StageConfig:
        return self._config
    
    def set_stage(self, stage: int):
        """Change current stage."""
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage {stage}. Must be 0-3.")
        self._stage = stage
        self._config = self.STAGES[stage]
    
    @property
    def mu_enabled(self) -> bool:
        return self._config.mu
    
    @property
    def temporal_enabled(self) -> bool:
        return self._config.temporal
    
    @property
    def k1_enabled(self) -> bool:
        return self._config.k1
    
    @property
    def graph_enabled(self) -> bool:
        return self._config.graph
    
    def __repr__(self) -> str:
        components = []
        if self._config.mu:
            components.append("MU")
        if self._config.temporal:
            components.append("TEMPORAL")
        if self._config.k1:
            components.append("K-1")
        if self._config.graph:
            components.append("Graph")
        return f"Stage {self._stage}: {' + '.join(components)}"

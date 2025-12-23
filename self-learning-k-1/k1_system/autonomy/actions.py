"""
Action definitions for Phase 2 autonomy system.

Actions represent autonomous decisions the system wants to make.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Action:
    """
    Represents an autonomous action the system wants to take.
    
    Attributes:
        type: Action type ('add_agent', 'delete_agent', 'tune_parameter', 'stop_training')
        param_name: Parameter name (for tune_parameter actions)
        param_value: New parameter value (for tune_parameter actions)
        node_id: Node ID (for add_agent/delete_agent actions)
    """
    type: str
    param_name: Optional[str] = None
    param_value: Optional[float] = None
    node_id: Optional[int] = None
    
    def __str__(self) -> str:
        if self.type == 'tune_parameter':
            return f"{self.type}({self.param_name}={self.param_value})"
        elif self.type in ('add_agent', 'delete_agent'):
            return f"{self.type}(node_id={self.node_id})"
        return self.type
    
    def __repr__(self) -> str:
        return self.__str__()


# Action type constants
ACTION_ADD_AGENT = 'add_agent'
ACTION_DELETE_AGENT = 'delete_agent'
ACTION_TUNE_PARAMETER = 'tune_parameter'
ACTION_STOP_TRAINING = 'stop_training'

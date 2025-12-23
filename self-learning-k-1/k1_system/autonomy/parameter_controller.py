"""
Parameter Controller for Phase 2 autonomous adjustment.

Manages autonomous parameter adjustment based on system performance.
Uses rule-based heuristics to adapt parameters dynamically.
"""

from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class SystemState:
    """
    Captures current system state for parameter adjustment decisions.
    """
    iteration: int
    accuracy: float
    loss: float
    validation_accuracy: float
    total_agents: int
    active_agents_pct: float
    avg_trust: float
    trust_variance: float
    cache_hit_rate: float
    rollback_count_last_10k: int
    iterations_since_improvement: int
    structural_ops_time_pct: float
    avg_error_magnitude: float
    improvement_last_5k: float
    structural_changes_last_10k: int
    cache_size: int

    def count_agents_below_trust(self, threshold: float, all_agents: List) -> int:
        """Count agents below trust threshold."""
        return sum(1 for agent in all_agents if agent.trust < threshold)


class ParameterController:
    """
    Manages autonomous parameter adjustment in Phase 2.
    Uses rule-based heuristics to adapt parameters based on system performance.
    """

    def __init__(self, initial_config: Dict):
        """
        Initialize parameter controller with Phase 1 parameters.

        Args:
            initial_config: Configuration from config_phase1.json
        """
        # Load Phase 1 parameters (these become adjustable in Phase 2)
        self.deletion_threshold = initial_config['pruning']['trust_threshold']
        self.top_k = initial_config['learning']['top_k']
        self.exploration_rate = initial_config['exploration']['initial_rate']
        self.merge_similarity_threshold = initial_config['merging']['similarity_threshold']
        self.prune_interval = initial_config['operations']['prune_interval']
        self.merge_interval = initial_config['operations']['merge_interval']
        self.reorganize_interval = initial_config['operations']['reorganize_interval']
        self.cache_threshold = initial_config['trust']['cache_threshold']
        self.learning_rate = initial_config['learning']['learning_rate']
        self.usage_threshold = initial_config['pruning']['usage_threshold']
        self.dormancy_threshold = initial_config['pruning']['dormancy_threshold']
        self.error_cluster_min_size = initial_config['growing']['error_cluster_min_size']
        self.success_threshold = initial_config['growing']['success_threshold']
        self.failure_threshold = initial_config['growing']['failure_threshold']
        self.validation_drop_threshold = initial_config['pruning']['validation_drop_threshold']
        self.max_agents = initial_config['structure']['max_total_agents']
        self.new_agent_initial_trust = initial_config['growing']['new_agent_initial_trust']

        self.phase_2_active = False
        self.adjustment_history: List[Dict] = []

        # Parameter bounds (safety limits)
        self.bounds = {
            'deletion_threshold': (0.1, 0.4),
            'top_k': (1, 10),
            'exploration_rate': (0.05, 0.5),
            'merge_similarity_threshold': (0.7, 0.95),
            'cache_threshold': (0.5, 0.9),
            'learning_rate': (0.0001, 0.01),
            'prune_interval': (5000, 50000),
            'merge_interval': (5000, 50000),
            'reorganize_interval': (25000, 100000),
        }

        # Track adjustment effectiveness
        self.adjustment_evaluation = []

    def activate_phase_2(self):
        """Called when iteration reaches phase_1_duration."""
        self.phase_2_active = True
        print("Parameter controller: Phase 2 active, autonomous adjustment enabled")

    def check_and_adjust(self, iteration: int, state: SystemState, all_agents: List = None):
        """
        Main adjustment logic - called every iteration in Phase 2.
        Examines system state and adjusts parameters using rule-based heuristics.

        Args:
            iteration: Current iteration
            state: Current system state
            all_agents: List of all agents (for some checks)
        """
        if not self.phase_2_active:
            return

        # Only adjust periodically (every 1000 iterations) to avoid instability
        if iteration % 1000 != 0:
            return

        # Run all adjustment rules
        self._adjust_deletion_threshold(state, all_agents or [])
        self._adjust_top_k(state)
        self._adjust_exploration_rate(state)
        self._adjust_merge_threshold(state)
        self._adjust_operation_intervals(state)
        self._adjust_cache_threshold(state)
        self._adjust_learning_rate(state)
        self._adjust_growth_parameters(state)

        # Log adjustments if any were made
        if len(self.adjustment_history) > 0 and \
           self.adjustment_history[-1]['iteration'] == iteration:
            self._print_adjustment_summary()

    def _adjust_deletion_threshold(self, state: SystemState, all_agents: List):
        """
        Adjust how aggressively we prune agents.

        Args:
            state: Current system state
            all_agents: List of all agents
        """
        # Rule 1: Over-pruning detection (too many rollbacks)
        if state.rollback_count_last_10k > 2:
            adjustment = +0.05  # More conservative
            self._apply_adjustment('deletion_threshold', adjustment,
                                 f'over-pruning detected ({state.rollback_count_last_10k} rollbacks)')

        # Rule 2: Too many dead agents
        if all_agents:
            dead_agent_count = sum(1 for a in all_agents if a.trust < 0.15)
            dead_agent_pct = dead_agent_count / len(all_agents) if all_agents else 0

            if dead_agent_pct > 0.3:  # >30% are useless
                adjustment = -0.03  # More aggressive pruning
                self._apply_adjustment('deletion_threshold', adjustment,
                                     f'{dead_agent_pct:.1%} agents have low trust')

        # Rule 3: Stable performance with few agents - can be more aggressive
        if state.iterations_since_improvement < 1000 and state.total_agents < 500:
            adjustment = -0.02
            self._apply_adjustment('deletion_threshold', adjustment,
                                 'performance good, few agents')

    def _adjust_top_k(self, state: SystemState):
        """
        Adjust how many agents to update per error.

        Args:
            state: Current system state
        """
        # Rule 1: High error complexity → need more agents
        if state.avg_error_magnitude > 0.7:
            if self.top_k < 7:
                self._apply_adjustment('top_k', +1, 'high error complexity')

        # Rule 2: Low error complexity → focus on specialists
        elif state.avg_error_magnitude < 0.3:
            if self.top_k > 2:
                self._apply_adjustment('top_k', -1, 'low error complexity')

        # Rule 3: Plateau → try updating more agents
        if state.iterations_since_improvement > 15000:
            if self.top_k < 8:
                self._apply_adjustment('top_k', +2, 'plateau - explore more')

    def _adjust_exploration_rate(self, state: SystemState):
        """
        Adjust exploration vs exploitation balance.

        Args:
            state: Current system state
        """
        # Rule 1: Performance plateau → explore more
        if state.iterations_since_improvement > 20000:
            adjustment = +0.15
            self._apply_adjustment('exploration_rate', adjustment,
                                 'long plateau - increase exploration')

        # Rule 2: Rapid improvement → exploit current strategy
        improvement_rate = state.improvement_last_5k / 5000 if state.improvement_last_5k else 0
        if improvement_rate > 0.01:  # 1% improvement per 100 iterations
            adjustment = -0.05
            self._apply_adjustment('exploration_rate', adjustment,
                                 'rapid improvement - exploit more')

        # Rule 3: Natural decay over time (mature system needs less exploration)
        if state.iteration > 100000:
            decay_adjustment = -0.01
            self._apply_adjustment('exploration_rate', decay_adjustment,
                                 'maturity decay')

    def _adjust_merge_threshold(self, state: SystemState):
        """
        Adjust how easily agents are merged.

        Args:
            state: Current system state
        """
        # Rule 1: Too many agents → easier merging
        if state.total_agents > 800:
            adjustment = -0.05  # Lower threshold = easier to merge
            self._apply_adjustment('merge_similarity_threshold', adjustment,
                                 f'{state.total_agents} agents - enable merging')

        # Rule 2: Too few agents → harder merging (preserve diversity)
        elif state.total_agents < 200:
            adjustment = +0.05  # Higher threshold = harder to merge
            self._apply_adjustment('merge_similarity_threshold', adjustment,
                                 f'only {state.total_agents} agents - preserve diversity')

    def _adjust_operation_intervals(self, state: SystemState):
        """
        Adjust how often structural operations run.

        Args:
            state: Current system state
        """
        # Rule 1: Structural operations taking too much time
        if state.structural_ops_time_pct > 0.15:  # >15% of time
            self.prune_interval = int(self.prune_interval * 1.3)
            self.merge_interval = int(self.merge_interval * 1.3)
            self._log_adjustment('prune_interval', self.prune_interval,
                               'structural ops overhead high')
            self._log_adjustment('merge_interval', self.merge_interval,
                               'structural ops overhead high')

        # Rule 2: Structure very stable → can do ops less frequently
        if state.structural_changes_last_10k < 5:
            self.prune_interval = int(self.prune_interval * 1.2)
            self._log_adjustment('prune_interval', self.prune_interval,
                               'structure stable')

    def _adjust_cache_threshold(self, state: SystemState):
        """
        Adjust trust threshold for cache admission.

        Args:
            state: Current system state
        """
        # Rule 1: Cache hit rate low → easier admission
        if state.cache_hit_rate < 0.3:
            adjustment = -0.05
            self._apply_adjustment('cache_threshold', adjustment,
                                 f'low cache hit rate ({state.cache_hit_rate:.1%})')

        # Rule 2: Cache too large → stricter admission
        if state.cache_size > 1000:
            adjustment = +0.05
            self._apply_adjustment('cache_threshold', adjustment,
                                 f'cache size large ({state.cache_size})')

    def _adjust_learning_rate(self, state: SystemState):
        """
        Adjust learning rate based on training progress.

        Args:
            state: Current system state
        """
        # Rule 1: Very long plateau → increase learning rate
        if state.iterations_since_improvement > 30000:
            adjustment = self.learning_rate * 0.1  # 10% increase
            self._apply_adjustment('learning_rate', adjustment,
                                 'very long plateau')

        # Rule 2: Loss oscillating → decrease learning rate
        # (Would need loss history to detect this - simplified here)
        if state.iteration > 50000 and state.iterations_since_improvement > 10000:
            adjustment = -self.learning_rate * 0.1  # 10% decrease
            self._apply_adjustment('learning_rate', adjustment,
                                 'possible oscillation')

    def _adjust_growth_parameters(self, state: SystemState):
        """
        Adjust parameters for agent creation.

        Args:
            state: Current system state
        """
        # Rule 1: Near capacity → make growth stricter
        if state.total_agents > 0.9 * self.max_agents:
            adjustment = +20  # Increase minimum cluster size
            self._apply_adjustment('error_cluster_min_size', adjustment,
                                 f'near capacity ({state.total_agents}/{self.max_agents})')

        # Rule 2: Few agents, room to grow → easier growth
        elif state.total_agents < 0.5 * self.max_agents:
            adjustment = -10  # Decrease minimum cluster size
            self._apply_adjustment('error_cluster_min_size', adjustment,
                                 f'room to grow ({state.total_agents}/{self.max_agents})')

    def _apply_adjustment(self, param_name: str, adjustment: float, reason: str):
        """
        Apply parameter adjustment with bounds checking and logging.

        Args:
            param_name: Name of parameter
            adjustment: Adjustment value (can be relative or absolute)
            reason: Reason for adjustment
        """
        if not hasattr(self, param_name):
            return

        old_value = getattr(self, param_name)

        # For integer parameters
        if param_name in ['top_k', 'prune_interval', 'merge_interval', 'reorganize_interval',
                         'error_cluster_min_size', 'usage_threshold', 'dormancy_threshold']:
            new_value = int(old_value + adjustment)
        else:
            new_value = old_value + adjustment

        # Apply bounds
        if param_name in self.bounds:
            min_val, max_val = self.bounds[param_name]
            new_value = max(min_val, min(max_val, new_value))

        # Only update if actually changed
        if new_value != old_value:
            setattr(self, param_name, new_value)
            self._log_adjustment(param_name, new_value, reason, adjustment)

    def _log_adjustment(self, param_name: str, new_value: Any, reason: str, change: Any = None):
        """
        Record parameter adjustment in history.

        Args:
            param_name: Parameter name
            new_value: New value
            reason: Reason for change
            change: Amount of change
        """
        entry = {
            'iteration': self.adjustment_history[-1]['iteration'] if self.adjustment_history else 0,
            'parameter': param_name,
            'new_value': new_value,
            'change': change,
            'reason': reason
        }

        # Append or update current iteration's entry
        if self.adjustment_history and self.adjustment_history[-1].get('iteration') == entry['iteration']:
            # Add to existing iteration log
            pass
        self.adjustment_history.append(entry)

    def _print_adjustment_summary(self):
        """Print recent adjustments."""
        if not self.adjustment_history:
            return

        current_iter = self.adjustment_history[-1]['iteration']
        recent = [a for a in self.adjustment_history if a['iteration'] == current_iter]

        if recent:
            print(f"\n📊 Parameter Adjustments at Iteration {current_iter}:")
            for adj in recent:
                change_str = f"{adj['change']:+.3f}" if isinstance(adj['change'], float) else f"{adj['change']:+d}" if adj['change'] else "updated"
                new_val_str = f"{adj['new_value']:.3f}" if isinstance(adj['new_value'], float) else str(adj['new_value'])
                print(f"   • {adj['parameter']}: {change_str} → {new_val_str}")
                print(f"     Reason: {adj['reason']}")

    def get_current_parameters(self) -> Dict:
        """
        Get current parameter values.

        Returns:
            Dictionary of current parameters
        """
        return {
            'deletion_threshold': self.deletion_threshold,
            'top_k': self.top_k,
            'exploration_rate': self.exploration_rate,
            'merge_similarity_threshold': self.merge_similarity_threshold,
            'prune_interval': self.prune_interval,
            'merge_interval': self.merge_interval,
            'reorganize_interval': self.reorganize_interval,
            'cache_threshold': self.cache_threshold,
            'learning_rate': self.learning_rate,
            'error_cluster_min_size': self.error_cluster_min_size,
            'success_threshold': self.success_threshold,
            'failure_threshold': self.failure_threshold
        }

    def get_adjustment_history(self) -> List[Dict]:
        """Get complete adjustment history."""
        return self.adjustment_history

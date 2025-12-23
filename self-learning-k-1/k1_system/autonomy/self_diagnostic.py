"""
Self-diagnostic system for the Self-Learning K-1 System.

Detects and corrects common problems autonomously.
"""

from typing import List, Dict, Optional
import numpy as np


class SelfDiagnostic:
    """
    Detects and diagnoses system problems autonomously.
    """

    def __init__(self):
        """Initialize diagnostic system."""
        self.diagnostic_history = []
        self.problems_detected = []

    def diagnose(self,
                iteration: int,
                state: 'SystemState',
                all_agents: List,
                recent_performance: List[float]) -> Dict:
        """
        Run diagnostic checks on system state.

        Args:
            iteration: Current iteration
            state: System state
            all_agents: List of all agents
            recent_performance: Recent performance history

        Returns:
            Dictionary with diagnostic results and recommendations
        """
        results = {
            'iteration': iteration,
            'problems': [],
            'recommendations': []
        }

        # Check 1: Stagnant performance
        if self._check_stagnant_performance(recent_performance):
            results['problems'].append('stagnant_performance')
            results['recommendations'].append('increase_exploration')
            results['recommendations'].append('create_new_specialists')

        # Check 2: Too many low-trust agents
        if self._check_too_many_dead_agents(all_agents):
            results['problems'].append('too_many_dead_agents')
            results['recommendations'].append('aggressive_pruning')

        # Check 3: Overloaded hierarchy
        if self._check_overloaded_hierarchy(all_agents, state):
            results['problems'].append('overloaded_hierarchy')
            results['recommendations'].append('reorganize_structure')

        # Check 4: Underutilized agents
        if self._check_underutilized_agents(all_agents):
            results['problems'].append('underutilized_agents')
            results['recommendations'].append('merge_similar_agents')

        # Check 5: High variance in trust
        if self._check_high_trust_variance(state):
            results['problems'].append('high_trust_variance')
            results['recommendations'].append('stabilize_learning')

        # Check 6: Cache ineffective
        if self._check_cache_ineffective(state):
            results['problems'].append('cache_ineffective')
            results['recommendations'].append('adjust_cache_threshold')

        # Log diagnostics
        if results['problems']:
            self.diagnostic_history.append(results)
            self.problems_detected.extend(results['problems'])

        return results

    def _check_stagnant_performance(self, recent_performance: List[float]) -> bool:
        """Check if performance has stagnated."""
        if len(recent_performance) < 1000:
            return False

        # Check if performance hasn't improved in last 1000 iterations
        recent = recent_performance[-1000:]
        improvement = max(recent) - min(recent)

        return improvement < 0.001

    def _check_too_many_dead_agents(self, all_agents: List) -> bool:
        """Check if too many agents have low trust."""
        if not all_agents:
            return False

        dead_count = sum(1 for agent in all_agents if agent.trust < 0.15)
        dead_pct = dead_count / len(all_agents)

        return dead_pct > 0.4  # More than 40% dead

    def _check_overloaded_hierarchy(self, all_agents: List, state: 'SystemState') -> bool:
        """Check if hierarchy structure is inefficient."""
        # Too many agents is one sign
        if state.total_agents > 900:
            return True

        # High structural operation time
        if state.structural_ops_time_pct > 0.2:
            return True

        return False

    def _check_underutilized_agents(self, all_agents: List) -> bool:
        """Check if many agents are rarely used."""
        if not all_agents:
            return False

        underutilized = sum(1 for agent in all_agents if agent.get_avg_activation() < 0.01)
        underutilized_pct = underutilized / len(all_agents)

        return underutilized_pct > 0.3  # More than 30% rarely used

    def _check_high_trust_variance(self, state: 'SystemState') -> bool:
        """Check if trust scores are highly variable."""
        return state.trust_variance > 0.1

    def _check_cache_ineffective(self, state: 'SystemState') -> bool:
        """Check if cache is not helping."""
        return state.cache_hit_rate < 0.2 and state.cache_size > 100

    def get_diagnostic_summary(self) -> Dict:
        """
        Get summary of all diagnostics run.

        Returns:
            Summary dictionary
        """
        if not self.diagnostic_history:
            return {
                'total_diagnostics': 0,
                'total_problems': 0
            }

        problem_counts = {}
        for diag in self.diagnostic_history:
            for problem in diag['problems']:
                problem_counts[problem] = problem_counts.get(problem, 0) + 1

        return {
            'total_diagnostics': len(self.diagnostic_history),
            'total_problems': len(self.problems_detected),
            'problem_counts': problem_counts,
            'most_common_problem': max(problem_counts.items(), key=lambda x: x[1])[0] if problem_counts else None
        }

    def apply_recommendations(self,
                            recommendations: List[str],
                            param_controller: 'ParameterController') -> Dict:
        """
        Apply recommended fixes.

        Args:
            recommendations: List of recommendation codes
            param_controller: Parameter controller to adjust

        Returns:
            Dictionary of actions taken
        """
        actions = {}

        for rec in recommendations:
            if rec == 'increase_exploration':
                param_controller.exploration_rate = min(0.5, param_controller.exploration_rate * 1.5)
                actions['exploration_rate'] = param_controller.exploration_rate

            elif rec == 'aggressive_pruning':
                param_controller.deletion_threshold = min(0.4, param_controller.deletion_threshold + 0.05)
                actions['deletion_threshold'] = param_controller.deletion_threshold

            elif rec == 'reorganize_structure':
                # Trigger reorganization (would be done by main loop)
                actions['trigger_reorganization'] = True

            elif rec == 'merge_similar_agents':
                param_controller.merge_similarity_threshold = max(0.7, param_controller.merge_similarity_threshold - 0.05)
                actions['merge_similarity_threshold'] = param_controller.merge_similarity_threshold

            elif rec == 'stabilize_learning':
                param_controller.learning_rate = param_controller.learning_rate * 0.8
                actions['learning_rate'] = param_controller.learning_rate

            elif rec == 'adjust_cache_threshold':
                param_controller.cache_threshold = max(0.5, param_controller.cache_threshold - 0.1)
                actions['cache_threshold'] = param_controller.cache_threshold

            elif rec == 'create_new_specialists':
                # Lower threshold for agent creation
                param_controller.error_cluster_min_size = max(50, param_controller.error_cluster_min_size - 20)
                actions['error_cluster_min_size'] = param_controller.error_cluster_min_size

        return actions

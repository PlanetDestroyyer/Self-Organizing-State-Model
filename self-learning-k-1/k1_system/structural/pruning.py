"""
Self-pruning system for the Self-Learning K-1 System.

Removes unused or low-performing agents to maintain efficiency.
"""

from typing import List, Tuple
import numpy as np
from ..core.agent import Agent
from ..core.hierarchy import Hierarchy


class PruningSystem:
    """
    Manages agent pruning based on trust, usage, and performance.
    """

    def __init__(self,
                 hierarchy: Hierarchy,
                 trust_threshold: float = 0.2,
                 usage_threshold: int = 50,
                 dormancy_threshold: int = 10000,
                 validation_drop_threshold: float = 0.05,
                 min_agents_per_manager: int = 3):
        """
        Initialize pruning system.

        Args:
            hierarchy: Hierarchy to prune
            trust_threshold: Minimum trust to keep agent
            usage_threshold: Minimum usage count to keep agent
            dormancy_threshold: Maximum iterations without use
            validation_drop_threshold: Max performance drop allowed
            min_agents_per_manager: Minimum children per manager
        """
        self.hierarchy = hierarchy
        self.trust_threshold = trust_threshold
        self.usage_threshold = usage_threshold
        self.dormancy_threshold = dormancy_threshold
        self.validation_drop_threshold = validation_drop_threshold
        self.min_agents_per_manager = min_agents_per_manager

        self.pruning_history = []

    def should_delete_agent(self, agent: Agent, current_iteration: int) -> Tuple[bool, str]:
        """
        Determine if an agent should be deleted.

        Args:
            agent: Agent to check
            current_iteration: Current iteration

        Returns:
            (should_delete, reason) tuple
        """
        # Never delete root
        if agent == self.hierarchy.root:
            return False, "Root agent"

        # Never delete if protected
        if agent.protected:
            return False, "Protected"

        # Check if only agent in domain
        if agent.is_only_agent_in_domain():
            return False, "Only agent in domain"

        # Check trust increasing
        if agent.trust_increasing():
            return False, "Trust increasing"

        # Check criteria
        criteria_met = []

        # Trust too low
        if agent.trust < self.trust_threshold:
            criteria_met.append("low_trust")

        # Usage too low
        if agent.usage_count_last_10k < self.usage_threshold:
            criteria_met.append("low_usage")

        # Dormant for too long
        if (current_iteration - agent.last_used) > self.dormancy_threshold:
            criteria_met.append("dormant")

        # Need all criteria to delete
        if len(criteria_met) >= 2:  # At least 2 criteria
            return True, ", ".join(criteria_met)

        return False, "Criteria not met"

    def find_pruning_candidates(self, current_iteration: int) -> List[Tuple[Agent, str]]:
        """
        Find all agents that are candidates for pruning.

        Args:
            current_iteration: Current iteration

        Returns:
            List of (agent, reason) tuples
        """
        candidates = []
        all_agents = self.hierarchy.get_all_agents()

        for agent in all_agents:
            should_delete, reason = self.should_delete_agent(agent, current_iteration)

            if should_delete:
                candidates.append((agent, reason))

        return candidates

    def validate_deletion(self,
                         agent: Agent,
                         validation_func,
                         baseline_perf: float) -> bool:
        """
        Validate that deleting an agent doesn't hurt performance.

        Args:
            agent: Agent to validate deletion for
            validation_func: Function to compute validation performance
            baseline_perf: Baseline performance before deletion

        Returns:
            True if safe to delete
        """
        # Temporarily remove agent
        parent = agent.parent
        children = list(agent.child_agents)

        # Remove from hierarchy
        if parent:
            parent.remove_child(agent)

        # Reassign children to parent if needed
        for child in children:
            if parent:
                parent.add_child(child)

        # Validate performance
        new_perf = validation_func()

        # Check performance drop
        perf_drop = (baseline_perf - new_perf) / baseline_perf if baseline_perf > 0 else 0

        # Restore agent temporarily for decision
        if parent:
            parent.add_child(agent)
        for child in children:
            if parent:
                parent.remove_child(child)
            agent.add_child(child)

        # Decision
        return perf_drop <= self.validation_drop_threshold

    def prune_agent(self, agent: Agent, reason: str, current_iteration: int):
        """
        Permanently delete an agent (with PyTorch optimizer cleanup).

        Args:
            agent: Agent to delete
            reason: Reason for deletion
            current_iteration: Current iteration
        """
        # Reassign children to parent
        parent = agent.parent
        if parent:
            for child in list(agent.child_agents):
                agent.remove_child(child)
                parent.add_child(child)

        # Remove from hierarchy
        self.hierarchy.remove_agent(agent)

        # Clean up optimizer state (PyTorch)
        # Note: weight_updater should be set by the system that uses pruning
        if hasattr(self, 'weight_updater') and hasattr(self.weight_updater, 'remove_agent_optimizer'):
            self.weight_updater.remove_agent_optimizer(agent.id)

        # Log deletion
        self.pruning_history.append({
            'iteration': current_iteration,
            'agent_id': agent.id,
            'specialty': agent.specialty,
            'trust': agent.trust,
            'reason': reason
        })

    def prune_agents(self,
                    current_iteration: int,
                    validation_func=None,
                    baseline_perf: float = None) -> dict:
        """
        Execute pruning cycle.

        Args:
            current_iteration: Current iteration
            validation_func: Optional validation function
            baseline_perf: Optional baseline performance

        Returns:
            Dictionary with pruning statistics
        """
        # Find candidates
        candidates = self.find_pruning_candidates(current_iteration)

        deleted = []
        protected = []
        failed_validation = []

        for agent, reason in candidates:
            # Check if deleting would violate minimum agents per manager
            if agent.parent and len(agent.parent.child_agents) <= self.min_agents_per_manager:
                protected.append(agent.id)
                continue

            # Validate deletion if function provided
            if validation_func and baseline_perf is not None:
                if not self.validate_deletion(agent, validation_func, baseline_perf):
                    failed_validation.append(agent.id)
                    # Protect for a while
                    agent.mark_protected(current_iteration + 10000)
                    continue

            # Delete agent
            self.prune_agent(agent, reason, current_iteration)
            deleted.append(agent.id)

        return {
            'candidates': len(candidates),
            'deleted': len(deleted),
            'protected': len(protected),
            'failed_validation': len(failed_validation),
            'deleted_ids': deleted
        }

    def get_top_performers(self, k: int = 20) -> List[Agent]:
        """
        Get top k% highest-trust agents.

        Args:
            k: Percentage (0-100)

        Returns:
            List of top agents
        """
        all_agents = self.hierarchy.get_all_agents()
        sorted_agents = sorted(all_agents, key=lambda a: a.trust, reverse=True)

        n = max(1, int(len(sorted_agents) * k / 100))
        return sorted_agents[:n]

    def update_thresholds(self,
                         trust_threshold: float = None,
                         usage_threshold: int = None,
                         dormancy_threshold: int = None):
        """
        Update pruning thresholds.

        Args:
            trust_threshold: New trust threshold
            usage_threshold: New usage threshold
            dormancy_threshold: New dormancy threshold
        """
        if trust_threshold is not None:
            self.trust_threshold = trust_threshold
        if usage_threshold is not None:
            self.usage_threshold = usage_threshold
        if dormancy_threshold is not None:
            self.dormancy_threshold = dormancy_threshold

    def get_pruning_statistics(self) -> dict:
        """
        Get statistics about pruning history.

        Returns:
            Dictionary of statistics
        """
        if not self.pruning_history:
            return {
                'total_pruned': 0,
                'avg_trust_at_pruning': 0.0
            }

        trust_values = [entry['trust'] for entry in self.pruning_history]

        return {
            'total_pruned': len(self.pruning_history),
            'avg_trust_at_pruning': np.mean(trust_values),
            'min_trust_at_pruning': np.min(trust_values),
            'max_trust_at_pruning': np.max(trust_values),
            'recent_prunings': len([e for e in self.pruning_history[-100:]])
        }

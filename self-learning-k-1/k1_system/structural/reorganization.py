"""
Self-reorganization system for the Self-Learning K-1 System.

Optimizes the hierarchical structure based on usage patterns.
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
from ..core.agent import Agent
from ..core.hierarchy import Hierarchy


class ReorganizationSystem:
    """
    Manages structural reorganization to optimize efficiency.
    """

    def __init__(self,
                 hierarchy: Hierarchy,
                 max_moves_per_cycle: float = 0.1):
        """
        Initialize reorganization system.

        Args:
            hierarchy: Hierarchy to reorganize
            max_moves_per_cycle: Max fraction of agents to move per cycle
        """
        self.hierarchy = hierarchy
        self.max_moves_per_cycle = max_moves_per_cycle

        self.coactivation_matrix = defaultdict(lambda: defaultdict(int))
        self.total_activations = 0
        self.reorganization_history = []

    def record_activation_pattern(self, activated_agents: List[Agent]):
        """
        Record which agents activated together.

        Args:
            activated_agents: List of agents activated in this forward pass
        """
        self.total_activations += 1

        # Record all pairs
        for i, agent_a in enumerate(activated_agents):
            for agent_b in activated_agents[i+1:]:
                self.coactivation_matrix[agent_a.id][agent_b.id] += 1
                self.coactivation_matrix[agent_b.id][agent_a.id] += 1

    def compute_cross_branch_communication(self) -> float:
        """
        Compute percentage of cross-branch activations.

        Returns:
            Cross-branch activation percentage
        """
        if self.total_activations == 0:
            return 0.0

        cross_branch = 0
        same_branch = 0

        for agent_a_id, coactivations in self.coactivation_matrix.items():
            agent_a = self.hierarchy.find_agent(agent_a_id)
            if agent_a is None:
                continue

            for agent_b_id, count in coactivations.items():
                agent_b = self.hierarchy.find_agent(agent_b_id)
                if agent_b is None:
                    continue

                # Check if they share a parent
                if self._share_parent(agent_a, agent_b):
                    same_branch += count
                else:
                    cross_branch += count

        total = cross_branch + same_branch
        if total == 0:
            return 0.0

        return cross_branch / total

    def _share_parent(self, agent_a: Agent, agent_b: Agent) -> bool:
        """Check if two agents share a parent."""
        if agent_a.parent is None or agent_b.parent is None:
            return False

        # Check immediate parent
        if agent_a.parent == agent_b.parent:
            return True

        # Check if they're in same branch (share grandparent)
        path_a = set(self.hierarchy.get_path_to_root(agent_a))
        path_b = set(self.hierarchy.get_path_to_root(agent_b))

        # If paths overlap (excluding root), they're in same branch
        overlap = path_a & path_b
        return len(overlap) > 1  # More than just root

    def find_optimal_parent_by_coactivation(self, agent: Agent) -> Agent:
        """
        Find optimal parent for an agent based on coactivation patterns.

        Args:
            agent: Agent to find parent for

        Returns:
            Optimal parent agent
        """
        # Get agents this one frequently coactivates with
        coactivations = self.coactivation_matrix.get(agent.id, {})

        if not coactivations:
            return agent.parent  # No data, keep current parent

        # Find which parent would minimize cross-branch communication
        parent_scores = defaultdict(int)

        for other_id, count in coactivations.items():
            other = self.hierarchy.find_agent(other_id)
            if other is None or other == agent:
                continue

            # Consider this agent's parent
            if other.parent is not None:
                parent_scores[other.parent.id] += count

        if not parent_scores:
            return agent.parent

        # Find best parent
        best_parent_id = max(parent_scores, key=parent_scores.get)
        best_parent = self.hierarchy.find_agent(best_parent_id)

        if best_parent is None:
            return agent.parent

        # Don't move if it would create cycle or violate depth
        if self.hierarchy._would_create_cycle(agent, best_parent):
            return agent.parent

        if self.hierarchy.get_depth(best_parent) + 1 >= self.hierarchy.max_depth:
            return agent.parent

        return best_parent

    def find_misplaced_agents(self) -> List[Tuple[Agent, Agent]]:
        """
        Find agents that should be moved based on coactivation.

        Returns:
            List of (agent, optimal_parent) tuples
        """
        misplaced = []
        all_agents = self.hierarchy.get_all_agents()

        for agent in all_agents:
            if agent == self.hierarchy.root:
                continue

            optimal_parent = self.find_optimal_parent_by_coactivation(agent)

            if optimal_parent != agent.parent and optimal_parent is not None:
                misplaced.append((agent, optimal_parent))

        return misplaced

    def find_frequent_cross_domain_pairs(self, threshold: int = 100) -> List[Tuple[Agent, Agent]]:
        """
        Find pairs of agents in different domains that frequently coactivate.

        Args:
            threshold: Minimum coactivations to consider

        Returns:
            List of (agent_a, agent_b) tuples
        """
        cross_domain_pairs = []

        for agent_a_id, coactivations in self.coactivation_matrix.items():
            agent_a = self.hierarchy.find_agent(agent_a_id)
            if agent_a is None:
                continue

            for agent_b_id, count in coactivations.items():
                if count < threshold:
                    continue

                agent_b = self.hierarchy.find_agent(agent_b_id)
                if agent_b is None:
                    continue

                # Check if in different domains
                if not self._share_parent(agent_a, agent_b):
                    cross_domain_pairs.append((agent_a, agent_b))

        return cross_domain_pairs

    def create_bridge_agent(self,
                           agent_a: Agent,
                           agent_b: Agent,
                           current_iteration: int) -> Agent:
        """
        Create a bridge agent to connect two domains.

        Args:
            agent_a: First agent
            agent_b: Second agent
            current_iteration: Current iteration

        Returns:
            Bridge agent
        """
        # Create bridge agent with combined specialty
        bridge = Agent(
            agent_id=f"bridge_{agent_a.id}_{agent_b.id}",
            agent_type='agent',
            specialty=f"Bridge: {agent_a.specialty} ↔ {agent_b.specialty}",
            input_dim=agent_a.input_dim,
            hidden_dim=(agent_a.hidden_dim + agent_b.hidden_dim) // 2,
            output_dim=agent_a.output_dim,
            initial_trust=0.3,
            creation_iteration=current_iteration
        )

        # Initialize weights as average
        for key in bridge.weights:
            if key in agent_a.weights and key in agent_b.weights:
                # Handle dimension mismatches by taking smaller dimensions
                w_a = agent_a.weights[key]
                w_b = agent_b.weights[key]

                if w_a.shape == w_b.shape:
                    bridge.weights[key] = 0.5 * (w_a + w_b)

        # Add to hierarchy under root
        self.hierarchy.add_agent(bridge, self.hierarchy.root)

        return bridge

    def split_manager(self, manager: Agent, current_iteration: int) -> Tuple[Agent, Agent]:
        """
        Split a busy manager into two managers.

        Args:
            manager: Manager to split
            current_iteration: Current iteration

        Returns:
            Tuple of (new_manager_1, new_manager_2)
        """
        children = list(manager.child_agents)

        if len(children) < 6:  # Need enough children to split
            return None, None

        # Split children in half (could use clustering for better split)
        mid = len(children) // 2
        group1 = children[:mid]
        group2 = children[mid:]

        # Create two new managers
        manager1 = Agent(
            agent_id=f"{manager.id}_split_1",
            agent_type='manager',
            specialty=f"{manager.specialty} - Group 1",
            input_dim=manager.input_dim,
            hidden_dim=manager.hidden_dim,
            output_dim=manager.output_dim,
            initial_trust=manager.trust,
            creation_iteration=current_iteration
        )

        manager2 = Agent(
            agent_id=f"{manager.id}_split_2",
            agent_type='manager',
            specialty=f"{manager.specialty} - Group 2",
            input_dim=manager.input_dim,
            hidden_dim=manager.hidden_dim,
            output_dim=manager.output_dim,
            initial_trust=manager.trust,
            creation_iteration=current_iteration
        )

        # Copy weights
        for key in manager.weights:
            manager1.weights[key] = manager.weights[key].copy()
            manager2.weights[key] = manager.weights[key].copy()

        # Reassign children
        for child in group1:
            manager.remove_child(child)
            manager1.add_child(child)

        for child in group2:
            manager.remove_child(child)
            manager2.add_child(child)

        # Add new managers to hierarchy
        parent = manager.parent
        if parent:
            self.hierarchy.add_agent(manager1, parent)
            self.hierarchy.add_agent(manager2, parent)

        # Remove old manager
        self.hierarchy.remove_agent(manager)

        return manager1, manager2

    def consolidate_manager(self, manager: Agent) -> bool:
        """
        Consolidate a rarely-used manager by moving children to parent.

        Args:
            manager: Manager to consolidate

        Returns:
            True if consolidated
        """
        if manager.parent is None:
            return False

        # Move all children to parent
        parent = manager.parent
        for child in list(manager.child_agents):
            manager.remove_child(child)
            parent.add_child(child)

        # Remove manager
        self.hierarchy.remove_agent(manager)

        return True

    def reorganize(self,
                  current_iteration: int,
                  validation_func=None,
                  baseline_perf: float = None) -> dict:
        """
        Execute reorganization cycle.

        Args:
            current_iteration: Current iteration
            validation_func: Optional validation function
            baseline_perf: Optional baseline performance

        Returns:
            Dictionary with reorganization statistics
        """
        results = {
            'agents_moved': 0,
            'bridges_created': 0,
            'managers_split': 0,
            'managers_consolidated': 0,
            'cross_branch_pct': 0.0
        }

        # Compute efficiency metrics
        cross_branch_pct = self.compute_cross_branch_communication()
        results['cross_branch_pct'] = cross_branch_pct

        # Find misplaced agents
        misplaced = self.find_misplaced_agents()

        # Move agents (limited to max_moves_per_cycle)
        total_agents = self.hierarchy.count_agents()
        max_moves = max(1, int(total_agents * self.max_moves_per_cycle))

        for agent, optimal_parent in misplaced[:max_moves]:
            # Move agent
            if self.hierarchy.move_agent(agent, optimal_parent):
                results['agents_moved'] += 1

        # Create bridges for frequent cross-domain patterns
        cross_domain_pairs = self.find_frequent_cross_domain_pairs()

        for agent_a, agent_b in cross_domain_pairs[:5]:  # Max 5 bridges per cycle
            bridge = self.create_bridge_agent(agent_a, agent_b, current_iteration)
            results['bridges_created'] += 1

        # Split busy managers
        managers = self.hierarchy.get_agents_by_type('manager')
        for manager in managers:
            if manager.get_avg_activation() > 0.8 and len(manager.child_agents) > 30:
                m1, m2 = self.split_manager(manager, current_iteration)
                if m1 and m2:
                    results['managers_split'] += 1

        # Consolidate rarely-used managers
        for manager in managers:
            if manager.get_avg_activation() < 0.1:
                if self.consolidate_manager(manager):
                    results['managers_consolidated'] += 1

        # Log reorganization
        self.reorganization_history.append({
            'iteration': current_iteration,
            'agents_moved': results['agents_moved'],
            'cross_branch_pct': cross_branch_pct
        })

        return results

    def get_reorganization_statistics(self) -> dict:
        """
        Get statistics about reorganization history.

        Returns:
            Dictionary of statistics
        """
        if not self.reorganization_history:
            return {
                'total_reorganizations': 0
            }

        return {
            'total_reorganizations': len(self.reorganization_history),
            'total_moves': sum([e['agents_moved'] for e in self.reorganization_history]),
            'avg_cross_branch_pct': np.mean([e['cross_branch_pct'] for e in self.reorganization_history])
        }

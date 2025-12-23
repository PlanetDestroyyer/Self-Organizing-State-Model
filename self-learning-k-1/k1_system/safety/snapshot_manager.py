"""
Snapshot and rollback system for the Self-Learning K-1 System.

Provides safety mechanism to revert structural changes if they hurt performance.
"""

import copy
from typing import List, Dict, Optional
from collections import deque
from ..core.hierarchy import Hierarchy
from ..core.agent import Agent


class StructureSnapshot:
    """
    Captures complete system state for rollback.
    """

    def __init__(self,
                 iteration: int,
                 hierarchy: Hierarchy,
                 performance_metrics: Dict):
        """
        Create a snapshot of current system state.

        Args:
            iteration: Current iteration
            hierarchy: Hierarchy to snapshot
            performance_metrics: Current performance metrics
        """
        self.iteration = iteration
        self.timestamp = iteration

        # Deep copy hierarchy structure
        self.hierarchy_structure = self._serialize_hierarchy(hierarchy)

        # Copy all agent states
        self.agent_states = {}
        for agent in hierarchy.get_all_agents():
            self.agent_states[agent.id] = self._serialize_agent(agent)

        # Performance metrics
        self.performance_metrics = copy.deepcopy(performance_metrics)

    def _serialize_hierarchy(self, hierarchy: Hierarchy) -> Dict:
        """
        Serialize hierarchy structure.

        Args:
            hierarchy: Hierarchy to serialize

        Returns:
            Serialized structure
        """
        return {
            'root_id': hierarchy.root.id if hierarchy.root else None,
            'max_depth': hierarchy.max_depth,
            'parent_child_map': self._build_parent_child_map(hierarchy)
        }

    def _build_parent_child_map(self, hierarchy: Hierarchy) -> Dict:
        """Build parent-child relationship map."""
        parent_child_map = {}

        for agent in hierarchy.get_all_agents():
            parent_child_map[agent.id] = {
                'parent_id': agent.parent.id if agent.parent else None,
                'children_ids': [child.id for child in agent.child_agents]
            }

        return parent_child_map

    def _serialize_agent(self, agent: Agent) -> Dict:
        """
        Serialize an agent's state.

        Args:
            agent: Agent to serialize

        Returns:
            Serialized agent state
        """
        return {
            'id': agent.id,
            'agent_type': agent.agent_type,
            'specialty': agent.specialty,
            'trust': agent.trust,
            'weights': copy.deepcopy(agent.weights),
            'creation_iteration': agent.creation_iteration,
            'last_used': agent.last_used,
            'usage_count': agent.usage_count,
            'success_count': agent.success_count,
            'failure_count': agent.failure_count,
            'total_error_reduction': agent.total_error_reduction,
            'protected': agent.protected,
            'protected_until': agent.protected_until,
            'input_dim': agent.input_dim,
            'hidden_dim': agent.hidden_dim,
            'output_dim': agent.output_dim
        }

    def restore(self, hierarchy: Hierarchy):
        """
        Restore hierarchy to this snapshot state.

        Args:
            hierarchy: Hierarchy to restore
        """
        # Clear current hierarchy
        current_agents = list(hierarchy.get_all_agents())
        for agent in current_agents:
            if agent != hierarchy.root:
                hierarchy.remove_agent(agent)

        # Recreate all agents
        agents_by_id = {}

        for agent_id, agent_state in self.agent_states.items():
            agent = self._deserialize_agent(agent_state)
            agents_by_id[agent_id] = agent

        # Restore hierarchy structure
        root_id = self.hierarchy_structure['root_id']
        if root_id and root_id in agents_by_id:
            hierarchy.set_root(agents_by_id[root_id])

        # Restore parent-child relationships
        parent_child_map = self.hierarchy_structure['parent_child_map']

        for agent_id, relationships in parent_child_map.items():
            agent = agents_by_id.get(agent_id)
            if not agent:
                continue

            # Set parent
            parent_id = relationships['parent_id']
            if parent_id and parent_id in agents_by_id:
                parent = agents_by_id[parent_id]
                hierarchy.add_agent(agent, parent)

        hierarchy.max_depth = self.hierarchy_structure['max_depth']

    def _deserialize_agent(self, agent_state: Dict) -> Agent:
        """
        Recreate an agent from serialized state.

        Args:
            agent_state: Serialized agent state

        Returns:
            Restored agent
        """
        agent = Agent(
            agent_id=agent_state['id'],
            agent_type=agent_state['agent_type'],
            specialty=agent_state['specialty'],
            input_dim=agent_state['input_dim'],
            hidden_dim=agent_state['hidden_dim'],
            output_dim=agent_state['output_dim'],
            initial_trust=agent_state['trust'],
            creation_iteration=agent_state['creation_iteration']
        )

        # Restore state
        agent.trust = agent_state['trust']
        agent.weights = copy.deepcopy(agent_state['weights'])
        agent.last_used = agent_state['last_used']
        agent.usage_count = agent_state['usage_count']
        agent.success_count = agent_state['success_count']
        agent.failure_count = agent_state['failure_count']
        agent.total_error_reduction = agent_state['total_error_reduction']
        agent.protected = agent_state['protected']
        agent.protected_until = agent_state['protected_until']

        return agent


class SnapshotManager:
    """
    Manages snapshots and rollback functionality.
    """

    def __init__(self,
                 max_snapshots: int = 5,
                 performance_drop_threshold: float = 0.15):
        """
        Initialize snapshot manager.

        Args:
            max_snapshots: Maximum number of snapshots to keep
            performance_drop_threshold: Performance drop to trigger rollback
        """
        self.max_snapshots = max_snapshots
        self.performance_drop_threshold = performance_drop_threshold

        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.rollback_history = []

    def create_snapshot(self,
                       iteration: int,
                       hierarchy: Hierarchy,
                       performance_metrics: Dict) -> StructureSnapshot:
        """
        Create and store a snapshot.

        Args:
            iteration: Current iteration
            hierarchy: Hierarchy to snapshot
            performance_metrics: Current performance

        Returns:
            Created snapshot
        """
        snapshot = StructureSnapshot(iteration, hierarchy, performance_metrics)
        self.snapshots.append(snapshot)
        return snapshot

    def should_rollback(self,
                       current_performance: float,
                       operation_type: str) -> bool:
        """
        Determine if should rollback based on performance.

        Args:
            current_performance: Current performance metric
            operation_type: Type of operation that was performed

        Returns:
            True if should rollback
        """
        if not self.snapshots:
            return False

        last_snapshot = self.snapshots[-1]
        snapshot_perf = last_snapshot.performance_metrics.get('accuracy', 0.0)

        # Compute performance drop
        if snapshot_perf > 0:
            perf_drop = (snapshot_perf - current_performance) / snapshot_perf
        else:
            perf_drop = 0.0

        # Rollback if drop exceeds threshold
        return perf_drop > self.performance_drop_threshold

    def rollback(self, hierarchy: Hierarchy) -> Optional[StructureSnapshot]:
        """
        Rollback to last snapshot.

        Args:
            hierarchy: Hierarchy to rollback

        Returns:
            Snapshot that was rolled back to, or None
        """
        if not self.snapshots:
            return None

        # Get last snapshot
        snapshot = self.snapshots[-1]

        # Restore hierarchy
        snapshot.restore(hierarchy)

        # Log rollback
        self.rollback_history.append({
            'iteration': snapshot.iteration,
            'performance': snapshot.performance_metrics
        })

        return snapshot

    def get_best_snapshot(self) -> Optional[StructureSnapshot]:
        """
        Get snapshot with best performance.

        Returns:
            Best snapshot or None
        """
        if not self.snapshots:
            return None

        best = max(self.snapshots,
                  key=lambda s: s.performance_metrics.get('accuracy', 0.0))
        return best

    def rollback_to_best(self, hierarchy: Hierarchy) -> Optional[StructureSnapshot]:
        """
        Rollback to best performing snapshot.

        Args:
            hierarchy: Hierarchy to rollback

        Returns:
            Best snapshot or None
        """
        best_snapshot = self.get_best_snapshot()

        if best_snapshot:
            best_snapshot.restore(hierarchy)
            self.rollback_history.append({
                'iteration': best_snapshot.iteration,
                'performance': best_snapshot.performance_metrics,
                'type': 'rollback_to_best'
            })

        return best_snapshot

    def get_rollback_count(self, window: int = 10000) -> int:
        """
        Get number of rollbacks in recent window.

        Args:
            window: Window size

        Returns:
            Rollback count
        """
        if not self.rollback_history:
            return 0

        if not self.snapshots:
            return len(self.rollback_history)

        # Count recent rollbacks
        latest_iteration = self.snapshots[-1].iteration
        count = sum(1 for rb in self.rollback_history
                   if latest_iteration - rb['iteration'] <= window)

        return count

    def get_snapshot_statistics(self) -> Dict:
        """
        Get statistics about snapshots.

        Returns:
            Dictionary of statistics
        """
        if not self.snapshots:
            return {
                'num_snapshots': 0,
                'num_rollbacks': 0
            }

        performances = [s.performance_metrics.get('accuracy', 0.0) for s in self.snapshots]

        return {
            'num_snapshots': len(self.snapshots),
            'num_rollbacks': len(self.rollback_history),
            'best_performance': max(performances),
            'worst_performance': min(performances),
            'latest_performance': performances[-1]
        }

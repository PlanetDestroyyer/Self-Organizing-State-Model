"""
Interpretability utilities for K-1 Self-Learning System.

Tracks agent activations, specializations, and routing decisions
to make the system interpretable and debuggable.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json


class AgentTracker:
    """
    Tracks agent activations and specializations over training.
    
    Key features:
    - Track which agents activate for which inputs
    - Identify agent specializations
    - Monitor agent trust evolution
    - Trace errors to specific agents
    """
    
    def __init__(self):
        # Agent activation tracking
        self.agent_activations = defaultdict(list)  # agent_id -> [(step, activation)]
        self.agent_inputs = defaultdict(list)  # agent_id -> [input_samples]
        self.agent_errors = defaultdict(list)  # agent_id -> [(step, error)]
        self.agent_updates = defaultdict(list)  # agent_id -> [step]
        self.agent_gradients = defaultdict(list)  # agent_id -> [(step, grad_norm)]
        
        # Routing decisions
        self.routing_decisions = []  # [(step, input_summary, agents_activated)]
        
        # Step tracking
        self.current_step = 0
        
    def record_activation(self, step: int, agent_id: str, activation_level: float,
                         input_summary: Optional[str] = None):
        """Record agent activation."""
        self.agent_activations[agent_id].append((step, activation_level))
        if input_summary:
            self.agent_inputs[agent_id].append(input_summary)
    
    def record_routing(self, step: int, input_summary: str, activated_agents: List[str]):
        """Record routing decision."""
        self.routing_decisions.append((step, input_summary, activated_agents))
    
    def record_update(self, step: int, agent_id: str):
        """Record that agent was updated."""
        self.agent_updates[agent_id].append(step)
    
    def record_gradient(self, step: int, agent_id: str, gradient_norm: float):
        """Record gradient magnitude for agent."""
        self.agent_gradients[agent_id].append((step, gradient_norm))
    
    def record_error(self, step: int, agent_id: str, error: float):
        """Record error contributed by agent."""
        self.agent_errors[agent_id].append((step, error))
    
    def get_agent_stats(self, agent_id: str) -> Dict:
        """Get statistics for a specific agent."""
        activations = self.agent_activations.get(agent_id, [])
        updates = self.agent_updates.get(agent_id, [])
        gradients = self.agent_gradients.get(agent_id, [])
        errors = self.agent_errors.get(agent_id, [])
        
        return {
            'total_activations': len(activations),
            'total_updates': len(updates),
            'avg_activation': np.mean([a for _, a in activations]) if activations else 0,
            'avg_gradient': np.mean([g for _, g in gradients]) if gradients else 0,
            'avg_error': np.mean([e for _, e in errors]) if errors else 0,
            'update_frequency': len(updates) / (self.current_step + 1) if self.current_step > 0 else 0,
        }
    
    def get_agent_specialization(self, agent_id: str) -> Dict:
        """
        Analyze what types of inputs this agent specializes in.
        Returns input patterns this agent handles.
        """
        # For now, return input samples
        # TODO: Add semantic clustering of inputs
        return {
            'agent_id': agent_id,
            'sample_inputs': self.agent_inputs.get(agent_id, [])[:10],  # Show first 10
            'total_inputs_seen': len(self.agent_inputs.get(agent_id, [])),
        }
    
    def identify_specialists(self, min_activations: int = 100) -> List[Tuple[str, Dict]]:
        """
        Identify specialist agents (high activation, focused inputs).
        Returns list of (agent_id, specialization_info) tuples.
        """
        specialists = []
        
        for agent_id in self.agent_activations.keys():
            stats = self.get_agent_stats(agent_id)
            if stats['total_activations'] >= min_activations:
                if stats['avg_activation'] > 0.5:  # Highly activated
                    spec_info = self.get_agent_specialization(agent_id)
                    spec_info.update(stats)
                    specialists.append((agent_id, spec_info))
        
        # Sort by activation frequency
        specialists.sort(key=lambda x: x[1]['avg_activation'], reverse=True)
        return specialists
    
    def identify_underutilized_agents(self, min_expected_activations: int = 10) -> List[str]:
        """Find agents that are rarely used."""
        underutilized = []
        
        for agent_id in self.agent_activations.keys():
            stats = self.get_agent_stats(agent_id)
            if stats['total_activations'] < min_expected_activations:
                underutilized.append(agent_id)
        
        return underutilized
    
    def get_error_attribution(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Attribute errors to agents.
        Returns top-k agents causing most error.
        """
        agent_error_totals = {}
        
        for agent_id, errors in self.agent_errors.items():
            if errors:
                agent_error_totals[agent_id] = sum(e for _, e in errors)
        
        # Sort by total error
        sorted_agents = sorted(agent_error_totals.items(), key=lambda x: x[1], reverse=True)
        return sorted_agents[:top_k]
    
    def save_report(self, filepath: str):
        """Save interpretability report to file."""
        report = {
            'total_steps': self.current_step,
            'total_agents_tracked': len(self.agent_activations),
            'specialists': [(aid, info) for aid, info in self.identify_specialists()],
            'underutilized': self.identify_underutilized_agents(),
            'error_attribution': self.get_error_attribution(top_k=10),
            'agent_stats': {
                agent_id: self.get_agent_stats(agent_id)
                for agent_id in self.agent_activations.keys()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def print_summary(self):
        """Print interpretability summary."""
        print("\n" + "="*70)
        print("INTERPRETABILITY SUMMARY")
        print("="*70)
        
        print(f"Total steps tracked: {self.current_step}")
        print(f"Total agents tracked: {len(self.agent_activations)}")
        
        # Specialists
        specialists = self.identify_specialists(min_activations=10)
        print(f"\nSpecialist Agents ({len(specialists)}):")
        for i, (agent_id, info) in enumerate(specialists[:5], 1):
            print(f"  {i}. {agent_id}")
            print(f"     Activations: {info['total_activations']}")
            print(f"     Avg activation: {info['avg_activation']:.3f}")
            print(f"     Updates: {info['total_updates']}")
        
        # Underutilized
        underutilized = self.identify_underutilized_agents()
        print(f"\nUnderutilized Agents ({len(underutilized)}):")
        for agent_id in underutilized[:5]:
            stats = self.get_agent_stats(agent_id)
            print(f"  - {agent_id}: {stats['total_activations']} activations")
        
        # Error attribution
        error_agents = self.get_error_attribution(top_k=5)
        print(f"\nTop Error-Causing Agents:")
        for i, (agent_id, total_error) in enumerate(error_agents, 1):
            print(f"  {i}. {agent_id}: {total_error:.4f} total error")
        
        print("="*70 + "\n")

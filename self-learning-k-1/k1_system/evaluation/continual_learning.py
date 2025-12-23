"""
Continual Learning Evaluation for K-1 System.

Tests the system's ability to learn new domains without forgetting old ones.
This is K-1's KEY ADVANTAGE over traditional backprop.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import json
from pathlib import Path


class ContinualLearningEvaluator:
    """
    Evaluates continual learning performance.
    
    Key metric: How much does the model forget when learning new tasks?
    - Baseline: Typically forgets 50-80% (catastrophic forgetting)
    - K-1: Should forget <20% (sparse updates preserve old agents)
    """
    
    def __init__(self, model, data_loaders: Dict[str, any], output_dir: str = "continual_results"):
        """
        Initialize evaluator.
        
        Args:
            model: K-1 or baseline model
            data_loaders: Dict of domain_name -> DataLoader
            output_dir: Directory to save results
        """
        self.model = model
        self.data_loaders = data_loaders
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Track performance over time
        self.performance_history = {
            domain: [] for domain in data_loaders.keys()
        }
        
        self.training_order = []
        
    def evaluate_domain(self, domain_name: str, num_batches: int = 20) -> Tuple[float, float]:
        """
        Evaluate performance on a domain.
        
        Returns:
            (loss, perplexity) tuple
        """
        if domain_name not in self.data_loaders:
            raise ValueError(f"Unknown domain: {domain_name}")
        
        data_loader = self.data_loaders[domain_name]
        
        # Use model's validation method if available
        if hasattr(self.model, 'forward_pass'):
            # K-1 model
            from k1_system.learning.hybrid_trainer import validate_k1_system
            
            loss, perplexity = validate_k1_system(
                self.model.forward_pass,
                data_loader,
                embedding=self.model.embedding if hasattr(self.model, 'embedding') else None,
                output_proj=self.model.output_proj if hasattr(self.model, 'output_proj') else None,
                num_batches=num_batches,
                vocab_size=data_loader.get_vocab_size()
            )
        else:
            # Baseline model
            total_loss = 0.0
            total_tokens = 0
            loss_fn = nn.CrossEntropyLoss()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            for _ in range(num_batches):
                try:
                    x_batch, y_batch = data_loader.get_batch('val', batch_size=8, return_tensors='pt')
                    
                    # Forward pass
                    logits = self.model(x_batch)  # (batch, seq_len, vocab_size)
                    
                    # Compute loss
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                    total_loss += loss.item() * y_batch.numel()
                    total_tokens += y_batch.numel()
                except:
                    continue
            
            if total_tokens > 0:
                loss = total_loss / total_tokens
                perplexity = np.exp(min(loss, 100.0))
            else:
                loss, perplexity = float('inf'), float('inf')
        
        return loss, perplexity
    
    def record_performance(self, domain_name: str, loss: float, perplexity: float):
        """Record performance on a domain at current point in training."""
        self.performance_history[domain_name].append({
            'step': len(self.training_order),
            'last_trained_on': self.training_order[-1] if self.training_order else None,
            'loss': loss,
            'perplexity': perplexity
        })
    
    def evaluate_all_domains(self) -> Dict[str, Tuple[float, float]]:
        """Evaluate on all domains and record results."""
        results = {}
        
        print("\n" + "="*70)
        print("Evaluating all domains...")
        print("="*70)
        
        for domain_name in self.data_loaders.keys():
            loss, perplexity = self.evaluate_domain(domain_name)
            results[domain_name] = (loss, perplexity)
            self.record_performance(domain_name, loss, perplexity)
            print(f"  {domain_name:20s} | Loss: {loss:.4f} | Perplexity: {perplexity:.2f}")
        
        print("="*70 + "\n")
        return results
    
    def compute_forgetting(self, domain_name: str) -> float:
        """
        Compute forgetting for a domain.
        
        Forgetting = (Best Performance - Current Performance) / Best Performance
        
        Returns:
            Forgetting rate (0.0 = no forgetting, 1.0 = complete forgetting)
        """
        history = self.performance_history[domain_name]
        if len(history) < 2:
            return 0.0
        
        # Find best performance (lowest loss)
        best_loss = min(h['loss'] for h in history)
        current_loss = history[-1]['loss']
        
        # Compute forgetting rate
        if best_loss == 0:
            return 0.0
        
        forgetting_rate = (current_loss - best_loss) / best_loss
        return max(0.0, forgetting_rate)  # Clip to [0, inf)
    
    def run_continual_learning_experiment(self, 
                                         training_order: List[str],
                                         steps_per_domain: int = 5000) -> Dict:
        """
        Run full continual learning experiment.
        
        Args:
            training_order: List of domain names to train on in order
            steps_per_domain: Training steps per domain
            
        Returns:
            Results dictionary with forgetting metrics
        """
        print("\n" + "="*70)
        print("CONTINUAL LEARNING EXPERIMENT")
        print("="*70)
        print(f"Training order: {' → '.join(training_order)}")
        print(f"Steps per domain: {steps_per_domain:,}")
        print("="*70 + "\n")
        
        # Initial evaluation (before any training)
        print("Initial evaluation (before training):")
        self.evaluate_all_domains()
        
        results = {
            'training_order': training_order,
            'steps_per_domain': steps_per_domain,
            'performance_timeline': [],
            'forgetting_metrics': {}
        }
        
        # Train on each domain sequentially
        for i, domain_name in enumerate(training_order, 1):
            print(f"\n{'='*70}")
            print(f"Training on Domain {i}/{len(training_order)}: {domain_name}")
            print(f"{'='*70}\n")
            
            self.training_order.append(domain_name)
            
            # Train on this domain
            data_loader = self.data_loaders[domain_name]
            
            if hasattr(self.model, 'train'):
                # K-1 model
                self.model.train(data=None, max_steps=steps_per_domain)
            else:
                # Baseline model - would need baseline trainer
                print("⚠️  Baseline training not implemented in evaluator")
            
            # Evaluate on ALL domains after training
            print(f"\nAfter training on {domain_name}:")
            domain_results = self.evaluate_all_domains()
            
            # Record timeline
            results['performance_timeline'].append({
                'after_training_on': domain_name,
                'results': domain_results
            })
        
        # Compute final forgetting metrics
        print("\n" + "="*70)
        print("FORGETTING ANALYSIS")
        print("="*70)
        
        for domain_name in self.data_loaders.keys():
            forgetting = self.compute_forgetting(domain_name)
            results['forgetting_metrics'][domain_name] = {
                'forgetting_rate': forgetting,
                'performance_history': self.performance_history[domain_name]
            }
            
            print(f"{domain_name:20s} | Forgetting: {forgetting*100:.1f}%")
        
        print("="*70 + "\n")
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict):
        """Save experiment results."""
        output_file = self.output_dir / "continual_learning_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {output_file}")
    
    def compare_models(self, baseline_results: Dict, k1_results: Dict):
        """
        Compare K-1 vs Baseline on continual learning.
        
        Args:
            baseline_results: Results from baseline model
            k1_results: Results from K-1 model
        """
        print("\n" + "="*70)
        print("CONTINUAL LEARNING COMPARISON: K-1 vs Baseline")
        print("="*70)
        
        print(f"\n{'Domain':<20} {'Baseline Forgetting':<20} {'K-1 Forgetting':<20} {'Improvement'}")
        print("-"*70)
        
        for domain in baseline_results['forgetting_metrics'].keys():
            baseline_forget = baseline_results['forgetting_metrics'][domain]['forgetting_rate']
            k1_forget = k1_results['forgetting_metrics'][domain]['forgetting_rate']
            
            improvement = (baseline_forget - k1_forget) / baseline_forget * 100 if baseline_forget > 0 else 0
            
            print(f"{domain:<20} {baseline_forget*100:>18.1f}% {k1_forget*100:>18.1f}% {improvement:>12.1f}%")
        
        # Overall statistics
        baseline_avg = np.mean([m['forgetting_rate'] for m in baseline_results['forgetting_metrics'].values()])
        k1_avg = np.mean([m['forgetting_rate'] for m in k1_results['forgetting_metrics'].values()])
        overall_improvement = (baseline_avg - k1_avg) / baseline_avg * 100 if baseline_avg > 0 else 0
        
        print("-"*70)
        print(f"{'AVERAGE':<20} {baseline_avg*100:>18.1f}% {k1_avg*100:>18.1f}% {overall_improvement:>12.1f}%")
        print("="*70)
        
        if overall_improvement > 30:
            print("\n✅ SUCCESS: K-1 shows significant continual learning advantage!")
        elif overall_improvement > 10:
            print("\n✓ GOOD: K-1 shows moderate continual learning advantage")
        else:
            print("\n⚠️  WEAK: K-1 advantage not clear - may need tuning")
        
        print()

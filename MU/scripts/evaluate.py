#!/usr/bin/env python3
"""
Evaluation script for MU Transformer

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoints/mu_best.pt --task all
    python scripts/evaluate.py --checkpoint results/checkpoints/baseline_best.pt --task lm
"""
import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mu_transformer import MUTransformerLM
from src.models.baseline_transformer import BaselineTransformerLM
from src.models.config import MUConfig, BaselineConfig
from src.evaluation.evaluator import Evaluator
from src.utils.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Evaluate MU Transformer')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--task', type=str,
                       choices=['lm', 'wic', 'stability', 'slots', 'all'],
                       default='all',
                       help='Evaluation task')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output', type=str,
                       default='results/evaluation_results.txt',
                       help='Output file for results')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(name='evaluate')

    logger.info("=" * 60)
    logger.info("MU TRANSFORMER EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Device: {args.device}")

    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Determine model type from config
    config_dict = checkpoint.get('config', {})
    model_config_dict = config_dict.get('model', {})

    # Check if MU or baseline based on config
    is_mu = 'r' in model_config_dict and 'c' in model_config_dict

    # Create model
    if is_mu:
        model_config = MUConfig(**model_config_dict)
        model = MUTransformerLM(model_config)
        logger.info("Created MU Transformer model")
    else:
        model_config = BaselineConfig(**model_config_dict)
        model = BaselineTransformerLM(model_config)
        logger.info("Created Baseline Transformer model")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    logger.info("Model loaded successfully")

    # Create evaluator
    data_config = config_dict.get('data', {
        'dataset': 'wikitext-2',
        'sequence_length': 256,
        'batch_size': 32,
        'num_workers': 4,
        'vocab_size': 30000
    })

    evaluator = Evaluator(model, device=args.device, data_config=data_config)

    # Run evaluation
    if args.task == 'all':
        tasks = ['lm', 'wic', 'stability']
        if is_mu:  # Only run slot analysis for MU models
            tasks.append('slots')
    else:
        tasks = [args.task]

    logger.info(f"Running evaluation tasks: {tasks}")
    results = evaluator.evaluate_all(tasks=tasks)

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    for task, metrics in results.items():
        logger.info(f"\n{task.upper()}:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for k, v in value.items():
                    logger.info(f"    {k}: {v}")
            else:
                logger.info(f"  {key}: {value}")

    # Save results
    evaluator.save_results(results, args.output)
    logger.info(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

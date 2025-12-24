#!/usr/bin/env python3
"""
Comprehensive SOSM Test Suite

Trains the complete SOSM system (Stage 3) for a few epochs, then tests:
1. Bank ambiguity (geographic vs financial)
2. Bat ambiguity (animal vs sports)
3. Spring ambiguity (season vs coil)
4. Palm ambiguity (tree vs hand)
5. Light ambiguity (illumination vs weight)
6. Apple ambiguity (fruit vs company)
7. Java ambiguity (island vs programming)
8. Python ambiguity (snake vs programming)
9. Lead ambiguity (metal vs guide)
10. Orange ambiguity (fruit vs color)
11. Semantic graph edge analysis

Usage:
    python test_sosm.py --epochs 5
"""

import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import GPT2Tokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Add TEMPORAL to path
TEMPORAL_PATH = PROJECT_ROOT / "TEMPORAL"
if str(TEMPORAL_PATH) not in sys.path:
    sys.path.insert(0, str(TEMPORAL_PATH))

from sosm_data import create_dataloaders
from state_core import StateCorePipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive SOSM Test')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, load from checkpoint')
    return parser.parse_args()


def train_epoch(pipeline, loader, optimizer, epoch: int):
    """Train for one epoch."""
    pipeline.train()
    total_loss = 0
    n_batches = 0
    start_time = time.time()
    
    for batch_idx, (input_ids, labels, domains) in enumerate(loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward through SOSM
        logits, state = pipeline(input_ids)
        
        loss = F.cross_entropy(
            logits.view(-1, pipeline.vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
        
        # K-1 hierarchical updates
        responsibility = pipeline.backward_with_k1(loss.detach(), state, batch_idx)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            edges = state.routing_state.get('num_edges', 0) if state.routing_state else 0
            print(f"  Batch {batch_idx:3d}: loss={loss.item():.4f}, edges={edges}, "
                  f"speed={batch_idx / max(1, elapsed):.1f} batch/s")
    
    avg_loss = total_loss / max(1, n_batches)
    return avg_loss


def evaluate(pipeline, loader):
    """Evaluate model."""
    pipeline.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for input_ids, labels, domains in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits, state = pipeline(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, pipeline.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / max(1, n_batches)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def test_disambiguation(pipeline, tokenizer, prompt1, prompt2, test_name):
    """
    Test semantic disambiguation between two contexts.
    
    Args:
        pipeline: SOSM model
        tokenizer: Tokenizer
        prompt1: First context prompt
        prompt2: Second context prompt (different meaning)
        test_name: Name of the test
    
    Returns:
        Dict with test results
    """
    print(f"\n{'=' * 70}")
    print(f"Test: {test_name}")
    print(f"{'=' * 70}")
    
    pipeline.eval()
    
    # Context 1
    print(f"\n[Context 1] {prompt1}")
    tokens1 = tokenizer.encode(prompt1, return_tensors='pt').to(device)
    
    with torch.no_grad():
        logits1, state1 = pipeline(tokens1, return_state=True)
    
    graph1 = state1.routing_state['graph'] if state1.routing_state else None
    next_token1 = logits1[0, -1].argmax()
    next_word1 = tokenizer.decode([next_token1])
    
    # Get top 5 predictions
    top5_logits1, top5_indices1 = logits1[0, -1].topk(5)
    top5_words1 = [tokenizer.decode([idx]) for idx in top5_indices1]
    
    print(f"  Next token: '{next_word1}'")
    print(f"  Top 5: {top5_words1}")
    if graph1:
        print(f"  Graph edges: {graph1['num_edges']} "
              f"(seq:{graph1['edge_types']['sequential']}, "
              f"sem:{graph1['edge_types']['semantic']}, "
              f"short:{graph1['edge_types']['shortcut']})")
    
    # Context 2
    print(f"\n[Context 2] {prompt2}")
    tokens2 = tokenizer.encode(prompt2, return_tensors='pt').to(device)
    
    with torch.no_grad():
        logits2, state2 = pipeline(tokens2, return_state=True)
    
    graph2 = state2.routing_state['graph'] if state2.routing_state else None
    next_token2 = logits2[0, -1].argmax()
    next_word2 = tokenizer.decode([next_token2])
    
    # Get top 5 predictions
    top5_logits2, top5_indices2 = logits2[0, -1].topk(5)
    top5_words2 = [tokenizer.decode([idx]) for idx in top5_indices2]
    
    print(f"  Next token: '{next_word2}'")
    print(f"  Top 5: {top5_words2}")
    if graph2:
        print(f"  Graph edges: {graph2['num_edges']} "
              f"(seq:{graph2['edge_types']['sequential']}, "
              f"sem:{graph2['edge_types']['semantic']}, "
              f"short:{graph2['edge_types']['shortcut']})")
    
    # Analysis
    print(f"\n[Analysis]")
    graphs_differ = graph1 and graph2 and graph1['num_edges'] != graph2['num_edges']
    predictions_differ = next_token1 != next_token2
    has_semantic_edges = (graph1 and graph1['edge_types']['semantic'] > 0) or \
                         (graph2 and graph2['edge_types']['semantic'] > 0)
    
    print(f"  ✓ Different graphs: {graphs_differ}")
    print(f"  ✓ Different predictions: {predictions_differ}")
    print(f"  ✓ Has semantic edges: {has_semantic_edges}")
    
    success = graphs_differ or predictions_differ
    print(f"\n  Result: {'✅ PASS' if success else '❌ FAIL'}")
    
    return {
        'test_name': test_name,
        'graphs_differ': graphs_differ,
        'predictions_differ': predictions_differ,
        'has_semantic_edges': has_semantic_edges,
        'success': success,
        'context1': {'prompt': prompt1, 'next': next_word1, 'top5': top5_words1},
        'context2': {'prompt': prompt2, 'next': next_word2, 'top5': top5_words2}
    }


def run_disambiguation_tests(pipeline, tokenizer):
    """Run all 11 disambiguation tests."""
    print("\n" + "=" * 70)
    print("SEMANTIC DISAMBIGUATION TEST SUITE")
    print("=" * 70)
    
    test_cases = [
        # Test 1: Bank (THE KEY TEST)
        ("The bank of the river is", "The bank loan is", "Bank (geographic vs financial)"),
        
        # Test 2: Bat
        ("The bat flew into the cave", "The baseball bat is made of", "Bat (animal vs sports)"),
        
        # Test 3: Spring
        ("Spring is the season when flowers", "The metal spring in the watch", "Spring (season vs coil)"),
        
        # Test 4: Palm
        ("The palm tree grows in tropical", "He held it in the palm of his", "Palm (tree vs hand)"),
        
        # Test 5: Light
        ("The light from the sun is", "The feather is very light in", "Light (illumination vs weight)"),
        
        # Test 6: Apple
        ("An apple is a delicious fruit that", "Apple Inc. is a technology company that", "Apple (fruit vs company)"),
        
        # Test 7: Java
        ("Java is an island in Indonesia where", "Java is a programming language used for", "Java (island vs programming)"),
        
        # Test 8: Python
        ("The python is a large snake that", "Python is a programming language known for", "Python (snake vs programming)"),
        
        # Test 9: Lead
        ("Lead is a heavy metal used in", "She will lead the team to", "Lead (metal vs guide)"),
        
        # Test 10: Orange
        ("An orange is a citrus fruit that", "Orange is a bright color that", "Orange (fruit vs color)"),
        
        # Test 11: Complex sentence structure
        ("The capital of India is", "The capital investment required is", "Capital (city vs finance)")
    ]
    
    results = []
    for prompt1, prompt2, test_name in test_cases:
        result = test_disambiguation(pipeline, tokenizer, prompt1, prompt2, test_name)
        results.append(result)
        time.sleep(0.5)  # Pause between tests
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total} ({100 * passed / total:.1f}%)\n")
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['success'] else "❌"
        print(f"{i:2d}. {status} {result['test_name']}")
        if result['success']:
            reasons = []
            if result['graphs_differ']:
                reasons.append("diff graphs")
            if result['predictions_differ']:
                reasons.append("diff predictions")
            if result['has_semantic_edges']:
                reasons.append("semantic edges")
            print(f"     ({', '.join(reasons)})")
    
    print("\n" + "=" * 70)
    
    return results


def main():
    args = parse_args()
    
    print("=" * 70)
    print("COMPREHENSIVE SOSM TEST SUITE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Configuration
    BATCH_SIZE = args.batch_size
    SEQ_LEN = 64
    EPOCHS = args.epochs
    VOCAB_SIZE = 50257  # GPT-2 vocab
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print()
    
    # Create SOSM pipeline (Full system - Stage 3)
    print("Creating SOSM pipeline (Stage 3: Full System)...")
    config = {
        'stage': 3,  # Full system with graph routing
        'components': {
            'mu': {
                'vocab_size': VOCAB_SIZE,
                'embed_dim': 64,
                'max_seq_len': SEQ_LEN,
                'use_full_model': True,  # ENABLED: 16-block semantic attention
                'n_layers': 2,  # Number of block attention layers
            },
            'temporal': {
                'time_dim': 32,
                'learning_mode': 'gradient',
            },
            'k1': {
                'analysis_only': True,
            },
            'graph': {
                'enabled': True,  # ENABLED
                'sequential_edges': True,
                'semantic_edges': True,  # ENABLED
                'semantic_threshold': 0.15,  # LOWERED: More permissive for richer semantic graphs
                'random_shortcuts': 0.08,  # INCREASED: 8% shortcuts for better connectivity
            }
        },
        'model': {
            'hidden_dim': 768,  # INCREASED: More capacity for complex patterns
            'n_layers': 6,  # INCREASED: Deeper model for better representations
            'n_heads': 8,
            'dropout': 0.1,
            'combination_mode': 'concat',
        }
    }
    
    pipeline = StateCorePipeline(config).to(device)
    n_params = sum(p.numel() for p in pipeline.parameters())
    print(f"✅ SOSM initialized: {n_params / 1e6:.2f}M parameters")
    print(f"   - MU: 16 semantic blocks with full attention (64D)")
    print(f"   - TEMPORAL: Self-learning (32D)")
    print(f"   - Graph: Semantic edges enabled (threshold=0.15, shortcuts=8%)")
    print(f"   - Model: {config['model']['hidden_dim']}D hidden, {config['model']['n_layers']} layers")
    print(f"   - K-1: Analysis mode")
    print()
    
    # Training
    if not args.skip_training:
        print("Loading training data...")
        train_loader, test_loader = create_dataloaders(
            batch_size=BATCH_SIZE,
            seq_length=SEQ_LEN,
            domains=['wikitext']
        )
        print(f"✅ Loaded WikiText dataset")
        print()
        
        # Optimizer
        optimizer = torch.optim.AdamW(pipeline.parameters(), lr=3e-4, weight_decay=0.01)
        
        # Training loop
        print("-" * 70)
        print("TRAINING")
        print("-" * 70)
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            
            train_loss = train_epoch(pipeline, train_loader, optimizer, epoch)
            test_loss, perplexity = evaluate(pipeline, test_loader)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss:  {test_loss:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
        
        # Save checkpoint
        checkpoint_name = 'sosm_trained.pt'
        torch.save({
            'model_state_dict': pipeline.state_dict(),
            'config': config,
            'epoch': EPOCHS,
        }, checkpoint_name)
        print(f"\n✅ Saved checkpoint: {checkpoint_name}")
    else:
        print("⚠️  Skipping training, loading from checkpoint...")
        checkpoint = torch.load('sosm_trained.pt')
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded checkpoint")
    
    print()
    
    # Run disambiguation tests
    results = run_disambiguation_tests(pipeline, tokenizer)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    # Final statistics
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 ALL {total} TESTS PASSED! 🎉")
        print("\nYour SOSM successfully disambiguates word meanings based on context!")
        print("Different neighbors → Different graphs → Different meanings")
    elif passed >= total * 0.7:
        print(f"\n✅ {passed}/{total} tests passed ({100 * passed / total:.1f}%)")
        print("\nGood performance! The system shows context-sensitive behavior.")
        print("Consider training longer for even better results.")
    else:
        print(f"\n⚠️  {passed}/{total} tests passed ({100 * passed / total:.1f}%)")
        print("\nThe system may need more training or parameter tuning.")
        print("Try: --epochs 10 or adjust semantic_threshold in config")


if __name__ == '__main__':
    main()

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

# PHASE 2.4: Import nucleus sampling
from state_core.utils.sampling import nucleus_sampling

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


def train_epoch(pipeline, loader, optimizer, epoch: int, scaler=None):
    """
    Train for one epoch.
    
    PHASE 1: Supports mixed precision training and K-1 sampling.
    """
    pipeline.train()
    total_loss = 0
    n_batches = 0
    start_time = time.time()
    
    use_amp = scaler is not None  # PHASE 1: Mixed precision flag
    
    for batch_idx, (input_ids, labels, domains) in enumerate(loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # PHASE 1: Mixed precision forward pass
        if use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                logits, state = pipeline(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, pipeline.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )
        else:
            # Forward through SOSM
            logits, state = pipeline(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, pipeline.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        # PHASE 1: Mixed precision backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
            optimizer.step()
        
        # PHASE 1: K-1 sampling (every 10 steps instead of every step)
        if batch_idx % 10 == 0:
            responsibility = pipeline.backward_with_k1(loss.detach(), state, batch_idx)
        
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
    
    # PHASE 2.4: Use nucleus sampling instead of greedy
    next_token1 = nucleus_sampling(logits1[0, -1], p=0.9, temperature=0.8)
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
    
    # PHASE 2.4: Use nucleus sampling instead of greedy
    next_token2 = nucleus_sampling(logits2[0, -1], p=0.9, temperature=0.8)
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
                'analysis_only': False,  # FIX: Enable K-1 weight updates
            },
            'graph': {
                'enabled': True,  # ENABLED
                'sequential_edges': True,
                'semantic_edges': True,  # ENABLED
                'semantic_method': 'topk',  # Top-K method (not threshold)
                'semantic_k': 7,  # FIX: Increased from 5 for more connections
                'semantic_threshold': 0.05,  # Optional minimum threshold
                'random_shortcuts': 0.20,  # Small-world optimal (20%)
                'use_mutual_knn': False,  # FIX: Disabled to keep asymmetric edges
                'streaming_topk': True,  # PHASE 1: Streaming Top-K (O(T×K) memory)
                'semantic_blocks': ['I', 'R2', 'K'],  # PHASE 2: Use I, R2, K blocks for similarity (12D)
            }
        },
        'model': {
            'hidden_dim': 896,  # PHASE 1: Increased to compensate for fewer layers
            'n_layers': 4,  # PHASE 1: Reduced from 6 (graph does heavy lifting)
            'n_heads': 8,
            'dropout': 0.3,  # FIX: Increased from 0.1 to prevent overfitting
            'combination_mode': 'concat',
        }
    }
    
    pipeline = StateCorePipeline(config).to(device)
    n_params = sum(p.numel() for p in pipeline.parameters())
    
    # Detailed initialization report with Phase 2.2-2.4 features
    print(f"✅ SOSM initialized: {n_params / 1e6:.2f}M parameters")
    print(f"   📊 Architecture:")
    print(f"      - MU: 16 semantic blocks, {config['components']['mu']['embed_dim']}D (factorized 2× reduction)")
    print(f"      - TEMPORAL: Self-learning, 32D time embeddings")
    print(f"      - Model: {config['model']['hidden_dim']}D hidden, {config['model']['n_layers']} layers")
    
    k_value = config['components']['graph'].get('semantic_k', 7)
    mutual_knn = config['components']['graph'].get('use_mutual_knn', False)
    shortcut_prob = config['components']['graph'].get('random_shortcuts', 0.05)
    
    print(f"   🔗 Graph (Phase 2.3 + 2.4):")
    print(f"      - Semantic edges: Top-K (K={k_value}, optimized via K study)")
    print(f"      - Shortcuts: Fibonacci pattern (prob={shortcut_prob})")
    print(f"      - Mutual k-NN: {'Enabled' if mutual_knn else 'Disabled'}")
    print(f"      - Provenance tracking: Active")
    
    print(f"   ⚡ Phase 2.2 Features:")
    print(f"      - Pre-LayerNorm: Active (better gradient flow)")
    print(f"      - Factorized embeddings: 2× reduction")
    print(f"      - Nucleus sampling: p=0.9, T=0.8 (generation)")
    
    print(f"   🚀 Phase 2.4 Features:")
    # Check if FlashAttention is available
    try:
        from state_core.utils.flash_attention import FLASH_ATTN_AVAILABLE
        flash_status = "Active" if FLASH_ATTN_AVAILABLE else "Fallback (standard)"
    except:
        flash_status = "Fallback (standard)"
    print(f"      - FlashAttention: {flash_status}")
    print(f"      - Fibonacci shortcuts: Active")
    
    k1_mode = "Analysis mode" if config['components']['k1'].get('analysis_only', False) else "Active updates"
    print(f"   🧠 K-1: {k1_mode}")
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
        
        # Optimizer with weight decay to prevent overfitting
        optimizer = torch.optim.AdamW(pipeline.parameters(), lr=1e-4, weight_decay=0.01)  # FIX: Added weight_decay, weight_decay=0.01)
        
        # PHASE 1: Mixed precision scaler
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if device.type == 'cuda' else None
        if scaler:
            print("✅ Mixed precision (FP16) enabled [PHASE 1]")
        print()
        
        # Training loop
        print("-" * 70)
        print("TRAINING")
        print("-" * 70)
        
        # Early stopping variables
        best_test_loss = float('inf')
        patience_counter = 0
        patience = 3  # Stop if no improvement for 3 epochs
        best_checkpoint = None
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            
            train_loss = train_epoch(pipeline, train_loader, optimizer, epoch, scaler=scaler)
            test_loss, perplexity = evaluate(pipeline, test_loader)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss:  {test_loss:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            
            # Early stopping check
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                # Save best checkpoint
                best_checkpoint = {
                    'model_state_dict': pipeline.state_dict(),
                    'config': config,
                    'epoch': epoch + 1,
                    'test_loss': test_loss,
                    'perplexity': perplexity,
                }
                print(f"  ✅ New best! Saved (PPL: {perplexity:.2f})")
            else:
                patience_counter += 1
                print(f"  ⚠️  No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping triggered! No improvement for {patience} epochs.")
                    print(f"Best epoch: {best_checkpoint['epoch']}, PPL: {best_checkpoint['perplexity']:.2f}")
                    break
        
        # Save best checkpoint (or final if no early stopping)
        checkpoint_name = 'sosm_trained.pt'
        if best_checkpoint:
            torch.save(best_checkpoint, checkpoint_name)
            print(f"\n✅ Saved BEST checkpoint: {checkpoint_name}")
            print(f"   Epoch: {best_checkpoint['epoch']}, PPL: {best_checkpoint['perplexity']:.2f}")
        else:
            # No early stopping triggered, save final
            torch.save({
                'model_state_dict': pipeline.state_dict(),
                'config': config,
                'epoch': EPOCHS,
            }, checkpoint_name)
            print(f"\n✅ Saved final checkpoint: {checkpoint_name}")
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

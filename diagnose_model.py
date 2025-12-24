#!/usr/bin/env python3
"""
Model Diagnostics - Check why generation is failing
"""

import torch
import torch.nn.functional as F
import math
from state_core import StateCorePipeline
from transformers import GPT2Tokenizer


def diagnose_model():
    """Run comprehensive diagnostics on SOSM model"""

    print("="*70)
    print("SOSM MODEL DIAGNOSTICS")
    print("="*70)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"   ✓ Vocab size: {len(tokenizer)}")

    # Create model (your current config)
    print("\n2. Loading model with CURRENT config...")
    config = {
        'stage': 3,
        'components': {
            'mu': {
                'vocab_size': 50257,
                'embed_dim': 64,  # YOUR CURRENT SIZE
                'use_full_model': True,
                'mu_layers': 2,
            },
            'temporal': {
                'time_dim': 32,  # YOUR CURRENT SIZE
            },
            'k1': {},
            'graph': {
                'sequential_edges': True,
                'semantic_edges': True,
                'random_shortcuts': 0.15,
            }
        },
        'model': {
            'hidden_dim': 256,  # YOUR CURRENT SIZE
            'n_layers': 6,
            'n_heads': 4,
            'dropout': 0.1
        }
    }

    model = StateCorePipeline(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Compare with GPT-2
    print("\n3. Comparison with GPT-2 Small:")
    gpt2_params = 124_000_000
    ratio = num_params / gpt2_params
    print(f"   GPT-2 Small: {gpt2_params:,} (124M)")
    print(f"   Your model:  {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"   Ratio: {ratio:.1%} of GPT-2")
    if ratio < 0.1:
        print(f"   ⚠️  WARNING: Your model is {1/ratio:.0f}x SMALLER than GPT-2!")

    # Test forward pass
    print("\n4. Testing forward pass...")
    test_text = "The capital of India is"
    input_ids = tokenizer.encode(test_text, return_tensors='pt').to(device)
    print(f"   Input: '{test_text}'")
    print(f"   Input shape: {input_ids.shape}")

    with torch.no_grad():
        logits, state = model(input_ids)

    print(f"   ✓ Output logits shape: {logits.shape}")
    print(f"   ✓ Semantic state shape: {state.semantic_state.shape}")
    if state.temporal_state is not None:
        print(f"   ✓ Temporal state shape: {state.temporal_state.shape}")

    # Check logit statistics
    print("\n5. Logit statistics:")
    print(f"   Min: {logits.min().item():.2f}")
    print(f"   Max: {logits.max().item():.2f}")
    print(f"   Mean: {logits.mean().item():.2f}")
    print(f"   Std: {logits.std().item():.2f}")

    # Check if logits are reasonable
    if abs(logits.mean().item()) > 10:
        print(f"   ⚠️  WARNING: Logits are too large/small (mean={logits.mean().item():.2f})")
    else:
        print(f"   ✓ Logits look reasonable")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    dummy_labels = input_ids.clone()
    logits, state = model(input_ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 50257),
                          dummy_labels[:, 1:].reshape(-1))

    loss.backward()

    # Check gradients
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats[name] = grad_norm

    # Find params with no gradients
    no_grad = [name for name, p in model.named_parameters()
               if p.requires_grad and p.grad is None]

    if no_grad:
        print(f"   ⚠️  {len(no_grad)} parameters have NO gradients!")
        print(f"      Example: {no_grad[:3]}")
    else:
        print(f"   ✓ All parameters have gradients")

    # Check for vanishing/exploding gradients
    if grad_stats:
        max_grad = max(grad_stats.values())
        min_grad = min(g for g in grad_stats.values() if g > 0)
        avg_grad = sum(grad_stats.values()) / len(grad_stats)

        print(f"   Gradient norms:")
        print(f"     Min: {min_grad:.6f}")
        print(f"     Max: {max_grad:.6f}")
        print(f"     Avg: {avg_grad:.6f}")

        if max_grad > 100:
            print(f"   ⚠️  WARNING: Gradients may be exploding (max={max_grad:.2f})")
        elif avg_grad < 1e-6:
            print(f"   ⚠️  WARNING: Gradients may be vanishing (avg={avg_grad:.2e})")
        else:
            print(f"   ✓ Gradients look reasonable")

    # Test overfit capacity
    print("\n7. Testing model capacity (can it memorize?)...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = None
    for step in range(100):
        optimizer.zero_grad()
        logits, state = model(input_ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 50257),
                              dummy_labels[:, 1:].reshape(-1))

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Final loss (after 100 steps): {final_loss:.4f}")
    print(f"   Reduction: {(initial_loss - final_loss):.4f}")

    if final_loss < initial_loss * 0.5:
        print(f"   ✓ Model CAN learn (loss reduced by {(1 - final_loss/initial_loss)*100:.1f}%)")
    else:
        print(f"   ⚠️  WARNING: Model struggling to learn (loss only reduced {(1 - final_loss/initial_loss)*100:.1f}%)")

    # Test generation
    print("\n8. Testing generation...")
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(test_text, return_tensors='pt').to(device)

        for _ in range(10):
            logits, _ = model(input_ids)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        output = tokenizer.decode(input_ids[0])
        print(f"   Input:  '{test_text}'")
        print(f"   Output: '{output}'")

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    issues = []
    if ratio < 0.1:
        issues.append(f"Model is {1/ratio:.0f}x smaller than GPT-2 Small")
    if num_params < 10_000_000:
        issues.append(f"Model has < 10M parameters (GPT-2 has 124M)")
    if no_grad:
        issues.append(f"{len(no_grad)} parameters not receiving gradients")
    if final_loss >= initial_loss * 0.5:
        issues.append("Model cannot memorize a single sentence")

    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✓ No major issues found!")

    print("\nRECOMMENDATIONS:")
    print("   1. Increase embed_dim: 64 → 512 or 768")
    print("   2. Increase hidden_dim: 256 → 1024 or 2048")
    print("   3. Start with Stage 0 (MU only) to verify training works")
    print("   4. Train for at least 10k steps to see learning")
    print("   5. Use train_fixed.py script for proper training")

    print("\n" + "="*70)


if __name__ == '__main__':
    diagnose_model()

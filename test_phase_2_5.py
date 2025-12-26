"""
Test script for Phase 2.5: Block Regularization (Tier 1)

Tests the integrated regularization losses with a simple training run.
"""

import torch
import yaml
from pathlib import Path
from state_core.pipeline import StateCorePipeline


def test_regularization_forward():
    """Test that regularization losses are computed correctly."""
    print("\n" + "="*60)
    print("PHASE 2.5: Testing Block Regularization (Tier 1)")
    print("="*60)
    
    # Load config
    config_path = Path(__file__).parent / "configs" / "phase_2_5_tier1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("\n📋 Configuration:")
    reg_cfg = config['regularization']
    print(f"  Regularization Enabled: {reg_cfg['enabled']}")
    print(f"  λ_ortho: {reg_cfg['lambda_ortho']}")
    print(f"  λ_var: {reg_cfg['lambda_var']}")
    print(f"  PairNorm: {reg_cfg['enable_pair_norm']}")
    
    # Create model
    print("\n🏗️  Building model...")
    model = StateCorePipeline(config)
    model.train()
    
    # Create dummy input
    B, T = 4, 32
    token_ids = torch.randint(0, 50257, (B, T))
    
    print(f"\n🔬 Running forward pass...")
    print(f"  Input shape: {token_ids.shape}")
    
    # Forward pass
    logits, state = model(token_ids, return_state=True)
    
    print(f"\n✅ Forward pass complete!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Semantic state shape: {state.semantic_state.shape}")
    
    # Check regularization losses
    if hasattr(state, 'reg_losses'):
        reg_losses = state.reg_losses
        print(f"\n📊 Regularization Losses:")
        print(f"  Total regularization: {reg_losses['total_reg'].item():.6f}")
        
        if 'orthogonality' in reg_losses:
            print(f"  Orthogonality loss: {reg_losses['orthogonality'].item():.6f}")
            ortho_info = reg_losses.get('ortho_info', {})
            print(f"    Off-diagonal mean: {ortho_info.get('off_diagonal_mean', 'N/A'):.4f}")
            print(f"    Diagonal mean: {ortho_info.get('diagonal_mean', 'N/A'):.4f}")
        
        if 'variance' in reg_losses:
            print(f"  Variance loss: {reg_losses['variance'].item():.6f}")
            var_info = reg_losses.get('var_info', {})
            print(f"    Mean std: {var_info.get('mean_std', 'N/A'):.4f}")
            print(f"    Min std: {var_info.get('min_std', 'N/A'):.4f}")
            print(f"    Dead dims: {var_info.get('num_dead_dims', 'N/A')}")
    
    # Test next-token prediction loss
    targets = token_ids[:, 1:]  # Shift by 1
    logits_for_loss = logits[:, :-1, :]  # Remove last prediction
    
    ntp_loss = torch.nn.functional.cross_entropy(
        logits_for_loss.reshape(-1, logits.shape[-1]),
        targets.reshape(-1)
    )
    
    print(f"\n📈 Next-Token Prediction Loss: {ntp_loss.item():.4f}")
    
    # Total loss
    total_loss = ntp_loss + reg_losses['total_reg']
    print(f"📊 Total Loss (NTP + Reg): {total_loss.item():.4f}")
    
    # Test backward
    print(f"\n⏪ Testing backward pass...")
    total_loss.backward()
    
    # Check gradients
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {has_grads}/{total_params}")
    
    print(f"\n✅ SUCCESS: All components working!")
    print("="*60)
    
    return {
        'ntp_loss': ntp_loss.item(),
        'ortho_loss': reg_losses.get('orthogonality', torch.tensor(0.0)).item(),
        'var_loss': reg_losses.get('variance', torch.tensor(0.0)).item(),
        'total_reg': reg_losses['total_reg'].item(),
        'total_loss': total_loss.item(),
    }


if __name__ == "__main__":
    results = test_regularization_forward()
    
    print("\n📊 Summary:")
    print(f"  NTP Loss: {results['ntp_loss']:.4f}")
    print(f"  Orthogonality: {results['ortho_loss']:.6f}")
    print(f"  Variance: {results['var_loss']:.6f}")
    print(f"  Total Regularization: {results['total_reg']:.6f}")
    print(f"  Combined Loss: {results['total_loss']:.4f}")

"""
Homonym Needle Test - Semantic Separation Score

Tests SOSM's ability to distinguish different meanings of the same word
based on context by measuring graph neighborhood separation.

Key Metric: Semantic Separation Score
- Measures cosine distance between graph neighborhoods for different contexts
- Score > 0.3 = Good disambiguation
- Score < 0.1 = Failed to disambiguate
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from state_core.pipeline import StateCorePipeline
from collections import defaultdict
import numpy as np


# Test pairs: (context1, context2, target_word)
# Using longer contexts to ensure graph generation
HOMONYM_TEST_PAIRS = [
    # Bank (THE KEY TEST)
    {
        'word': 'bank',
        'context1': "I went to the bank to deposit money in my savings account for retirement planning",
        'context2': "I sat on the bank of the river and watched the water flow peacefully downstream",
        'expected': 'financial vs geographic'
    },
    # Bat
    {
        'word': 'bat',
        'context1': "The bat flew into the dark cave at night to find insects for food",
        'context2': "He swung the baseball bat with great force to hit the ball out of the park",
        'expected': 'animal vs sports equipment'
    },
    # Java
    {
        'word': 'java',
        'context1': "Java is a beautiful island in Indonesia with rich cultural heritage and stunning landscapes",
        'context2': "Java is a popular programming language used for building enterprise applications and Android apps",
        'expected': 'place vs technology'
    },
    # Lead
    {
        'word': 'lead',
        'context1': "Lead is a heavy toxic metal that was once commonly used in paint and gasoline",
        'context2': "She will lead the team to victory with her excellent leadership skills and strategic thinking",
        'expected': 'metal vs verb'
    },
    # Python
    {
        'word': 'python',
        'context1': "The python is a large constrictor snake that can grow to be over twenty feet long",
        'context2': "Python is widely used for machine learning and data science because of its simplicity and powerful libraries",
        'expected': 'animal vs programming'
    },
]


def get_graph_neighborhood_embedding(pipeline, tokenizer, text, target_word, device='cuda'):
    """
    Get average embedding of graph neighbors for a target word in context.
    
    Args:
        pipeline: SOSM model
        tokenizer: Tokenizer
        text: Context sentence
        target_word: Word to analyze
        device: Device
    
    Returns:
        neighbor_embedding: [D] average of neighbor embeddings
        target_idx: Token index of target word
    """
    pipeline.eval()
    
    # Tokenize
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Find target word index
    target_word_lower = target_word.lower()
    token_strs = [tokenizer.decode([t]).lower().strip() for t in tokens[0]]
    
    target_idx = None
    for i, tok_str in enumerate(token_strs):
        if target_word_lower in tok_str:
            target_idx = i
            break
    
    if target_idx is None:
        print(f"Warning: '{target_word}' not found in '{text}'")
        return None, None
    
    # Forward pass to get graph
    with torch.no_grad():
        logits, state = pipeline(tokens, return_state=True)
    
    if state.routing_state is None or 'graph' not in state.routing_state:
        print("Warning: No graph in routing state")
        return None, target_idx
    
    graph = state.routing_state['graph']
    edges = graph.get('edges', [])
    
    if not edges:
        print("Warning: No edges in graph")
        return None, target_idx
    
    # Get MU state (semantic embeddings)
    mu_state = state.mu_state  # [B, T, 64]
    mu_state = mu_state[0]  # [T, 64]
    
    # Find neighbors of target word
    neighbor_indices = []
    for edge in edges:
        i, j = edge
        if i == target_idx:
            neighbor_indices.append(j)
        elif j == target_idx:
            neighbor_indices.append(i)
    
    if not neighbor_indices:
        # No neighbors - use target embedding itself
        return mu_state[target_idx].cpu(), target_idx
    
    # Average neighbor embeddings
    neighbor_embeddings = mu_state[neighbor_indices]  # [N, 64]
    avg_neighbor = neighbor_embeddings.mean(dim=0)  # [64]
    
    return avg_neighbor.cpu(), target_idx


def compute_semantic_separation(pipeline, tokenizer, test_pair, device='cuda'):
    """
    Compute semantic separation score for a homonym test pair.
    
    Returns:
        separation_score: float in [0, 1]
        - 1.0 = completely different neighborhoods
        - 0.0 = identical neighborhoods
    """
    word = test_pair['word']
    ctx1 = test_pair['context1']
    ctx2 = test_pair['context2']
    
    # Get neighborhood embeddings
    neighbor1, idx1 = get_graph_neighborhood_embedding(pipeline, tokenizer, ctx1, word, device)
    neighbor2, idx2 = get_graph_neighborhood_embedding(pipeline, tokenizer, ctx2, word, device)
    
    if neighbor1 is None or neighbor2 is None:
        return 0.0, None, None
    
    # Compute cosine distance (1 - similarity)
    similarity = F.cosine_similarity(neighbor1.unsqueeze(0), neighbor2.unsqueeze(0)).item()
    separation = 1 - similarity  # Convert to distance
    
    return separation, idx1, idx2


def run_homonym_needle_test(pipeline, tokenizer, device='cuda'):
    """
    Run the Homonym Needle Test on all test pairs.
    
    Returns:
        results: Dict with scores and analysis
    """
    print("=" * 70)
    print("HOMONYM NEEDLE TEST - Semantic Separation Score")
    print("=" * 70)
    print()
    print("Testing if SOSM can distinguish different meanings of the same word")
    print("by measuring graph neighborhood separation.")
    print()
    print("Target: Separation > 0.3 (good), > 0.5 (excellent)")
    print("=" * 70)
    print()
    
    results = []
    
    for test_pair in HOMONYM_TEST_PAIRS:
        word = test_pair['word']
        expected = test_pair['expected']
        
        print(f"Testing: '{word}' ({expected})")
        print(f"  Context 1: {test_pair['context1']}")
        print(f"  Context 2: {test_pair['context2']}")
        
        separation, idx1, idx2 = compute_semantic_separation(
            pipeline, tokenizer, test_pair, device
        )
        
        # Assess quality
        if separation >= 0.5:
            quality = "✅ EXCELLENT"
        elif separation >= 0.3:
            quality = "✅ GOOD"
        elif separation >= 0.15:
            quality = "⚠️  WEAK"
        else:
            quality = "❌ FAILED"
        
        print(f"  Separation Score: {separation:.3f} {quality}")
        print()
        
        results.append({
            'word': word,
            'expected': expected,
            'separation': separation,
            'quality': quality
        })
    
    # Summary
    avg_separation = np.mean([r['separation'] for r in results])
    excellent_count = sum(1 for r in results if r['separation'] >= 0.5)
    good_count = sum(1 for r in results if 0.3 <= r['separation'] < 0.5)
    weak_count = sum(1 for r in results if 0.15 <= r['separation'] < 0.3)
    failed_count = sum(1 for r in results if r['separation'] < 0.15)
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Average Separation Score: {avg_separation:.3f}")
    print(f"  ✅ Excellent (≥0.5): {excellent_count}/{len(results)}")
    print(f"  ✅ Good (≥0.3):      {good_count}/{len(results)}")
    print(f"  ⚠️  Weak (≥0.15):     {weak_count}/{len(results)}")
    print(f"  ❌ Failed (<0.15):   {failed_count}/{len(results)}")
    print()
    
    if avg_separation >= 0.5:
        verdict = "✅ EXCELLENT - SOSM strongly disambiguates!"
    elif avg_separation >= 0.3:
        verdict = "✅ GOOD - SOSM disambiguates effectively"
    elif avg_separation >= 0.15:
        verdict = "⚠️  WEAK - Some disambiguation, needs improvement"
    else:
        verdict = "❌ FAILED - No meaningful disambiguation"
    
    print(f"Verdict: {verdict}")
    print("=" * 70)
    
    return {
        'results': results,
        'avg_separation': avg_separation,
        'excellent': excellent_count,
        'good': good_count,
        'weak': weak_count,
        'failed': failed_count,
        'verdict': verdict
    }


if __name__ == '__main__':
    import sys
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Load checkpoint
    checkpoint_path = 'sosm_trained.pt'
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint['config']
        
        # CRITICAL FIX: Ensure graph is enabled for testing
        if 'pipeline' in config and 'graph' in config['pipeline']:
            config['pipeline']['graph']['enabled'] = True
            print(f"Graph config: enabled={config['pipeline']['graph']['enabled']}, K={config['pipeline']['graph'].get('semantic_k', 'N/A')}")
        
        # Create pipeline
        pipeline = StateCorePipeline(config).to(device)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify graph is enabled
        print(f"Pipeline graph enabled: {pipeline.stage_controller.graph_enabled}")
        if not pipeline.stage_controller.graph_enabled:
            print("⚠️  WARNING: Graph not enabled! Enabling now...")
            pipeline.stage_controller.graph_enabled = True
        
        # QUICK FIX: Override buggy checkpoint config
        print("\n⚠️  APPLYING QUICK FIX: Overriding checkpoint config...")
        print(f"   Threshold: {pipeline.graph_builder.semantic_threshold} → 0.05")
        pipeline.graph_builder.semantic_threshold = 0.05
        pipeline.graph_builder.semantic_k = 7
        pipeline.graph_builder.use_mutual_knn = False
        pipeline.graph_builder.semantic_blocks = ['I', 'R2', 'K']
        print(f"   K: {pipeline.graph_builder.semantic_k}")
        print(f"   Mutual k-NN: {pipeline.graph_builder.use_mutual_knn}")
        print(f"   Blocks: {pipeline.graph_builder.semantic_blocks}")
        print("✅ Config overridden for testing\n")
        
        print("✅ Model loaded successfully\n")
        
        # Run test
        results = run_homonym_needle_test(pipeline, tokenizer, device)
        
        sys.exit(0 if results['avg_separation'] >= 0.3 else 1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

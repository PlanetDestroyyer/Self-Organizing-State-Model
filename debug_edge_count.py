"""
Debug script to investigate why edge count is ~1200 instead of expected ~1800+

Analyzes:
1. How many semantic edges are created per token
2. Whether mutual k-NN is actually disabled
3. What's limiting edge creation
"""

import torch
from state_core.graph.graph_builder import GraphBuilder

# Create graph builder with same config as training
graph_builder = GraphBuilder(
    enable_sequential=True,
    enable_semantic=True,
    enable_shortcuts=True,
    semantic_k=7,
    semantic_method='topk',
    shortcut_prob=0.20,
    use_mutual_knn=False,  # Should be disabled!
    streaming_topk=True,
    semantic_blocks=['I', 'R2', 'K']
)

# Simulate a sequence
seq_len = 175  # Average from training
batch_size = 1

# Random semantic state (simulating MU output)
semantic_state = torch.randn(batch_size, seq_len, 64)  # Full 64D

# Build graph
graph = graph_builder.build_graph(
    seq_len=seq_len,
    semantic_state=semantic_state,
    batch_size=batch_size
)

print("=" * 70)
print("EDGE COUNT DEBUG")
print("=" * 70)
print(f"Sequence length: {seq_len}")
print(f"Semantic K: {graph_builder.semantic_k}")
print(f"Mutual k-NN: {graph_builder.use_mutual_knn}")
print(f"Bidirectional: {graph_builder.bidirectional}")
print(f"Shortcut prob: {graph_builder.shortcut_prob}")
print()

print("Edge counts:")
print(f"  Sequential: {graph['edge_types']['sequential']}")
print(f"  Semantic:   {graph['edge_types']['semantic']}")
print(f"  Shortcut:   {graph['edge_types']['shortcut']}")
print(f"  TOTAL:      {graph['num_edges']}")
print()

# Expected calculations
expected_sequential = (seq_len - 1) * 2  # bidirectional
expected_shortcuts_approx = int(seq_len * graph_builder.shortcut_prob * 2)  # bidirectional
print("Expected:")
print(f"  Sequential: {expected_sequential}")
print(f"  Shortcuts:  ~{expected_shortcuts_approx}")
print()

# Semantic edge analysis
semantic_edges = graph['edge_types']['semantic']
expected_semantic_unidirectional = seq_len * graph_builder.semantic_k
expected_semantic_bidirectional = expected_semantic_unidirectional * 2

print("Semantic edge analysis:")
print(f"  Expected (K={graph_builder.semantic_k}, unidirectional): {expected_semantic_unidirectional}")
print(f"  Expected (bidirectional): {expected_semantic_bidirectional}")
print(f"  Actual: {semantic_edges}")
print(f"  Ratio: {semantic_edges / expected_semantic_bidirectional:.2%}")
print()

if semantic_edges < expected_semantic_bidirectional:
    possible_reasons = []
    
    # Check if threshold filtering
    if graph_builder.semantic_threshold > 0:
        possible_reasons.append(f"Semantic threshold ({graph_builder.semantic_threshold}) filtering low similarities")
    
    # Check if mutual k-NN is somehow active
    if semantic_edges < expected_semantic_unidirectional:
        possible_reasons.append("Mutual k-NN might still be filtering (despite config)")
    
    # Check blockwise similarity impact
    if graph_builder.semantic_blocks != list(range(16)):
        possible_reasons.append(f"Blockwise similarity ({graph_builder.semantic_blocks}) may reduce matches")
    
    print("Possible reasons for low edge count:")
    for i, reason in enumerate(possible_reasons, 1):
        print(f"  {i}. {reason}")
else:
    print("✅ Edge count matches expectations!")

print("=" * 70)

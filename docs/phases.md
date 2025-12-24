# SOSM Implementation Phases: Complete Roadmap 🗺️

**5-Phase Plan**: From Quick Wins to Advanced Features  
**Timeline**: 6-8 weeks  
**Last Updated**: 2025-12-24

---

## 📋 Phase Overview

| Phase | Focus | Duration | Risk | Expected Gain |
|-------|-------|----------|------|---------------|
| **Phase 1** | Quick Wins | Week 1 | Very Low | 45% speed, 30% memory |
| **Phase 2** | Quality & Interpretability | Week 2 | Low | +8-10% accuracy |
| **Phase 3** | Scale & Advanced Features | Week 3-4 | Medium | 2× speed, 50% memory |
| **Phase 4** | Long-Range & Efficiency | Week 5-6 | Medium | Infinite context, 6× params |
| **Phase 5** | Production & Deployment | Week 7-8 | Low | Multi-GPU, serving |

---

# Phase 1: Quick Wins ⚡

**Timeline**: Week 1 (5-7 days)  
**Goal**: 45% speedup, 30% memory reduction, minimal risk  
**Risk Level**: 🟢 Very Low

## Items

### 1.1 Streaming Top-K (Replace Materialized Matrix)
**Priority**: 🔴 CRITICAL  
**Time**: 3-4 hours  
**Complexity**: Medium

**What**: Compute Top-K without creating full T×T similarity matrix

**Code**:
```python
def streaming_topk(semantic_state, K):
    T = semantic_state.size(0)
    state_norm = F.normalize(semantic_state, dim=-1)
    
    edges = []
    for i in range(T):
        query = state_norm[i]
        sims = state_norm @ query  # [T] - one row only
        
        # Mask self and adjacent
        sims[i] = -inf
        if i > 0: sims[i-1] = -inf
        if i < T-1: sims[i+1] = -inf
        
        # Top-K
        topk_idx = sims.topk(min(K, T-3)).indices
        edges.extend([(i, int(j)) for j in topk_idx])
    
    return edges
```

**Benefits**:
- ✅ Memory: O(T×K) instead of O(T²)
- ✅ Speed: 30-40% faster for large T
- ✅ Streaming-friendly

**File**: `state_core/graph/graph_builder.py`  
**Test**: Verify same edges as current method

---

### 1.2 Mutual k-NN Filtering
**Priority**: 🟡 HIGH  
**Time**: 1-2 hours  
**Complexity**: Low

**What**: Keep only bidirectional edges (i→j AND j→i)

**Code**:
```python
def mutual_knn_filter(edges):
    edge_set = set(edges)
    mutual = []
    
    for (i, j) in edges:
        if (j, i) in edge_set and i < j:
            mutual.append((i, j))
    
    return mutual
```

**Benefits**:
- ✅ Reduces hub tokens
- ✅ 20-30% fewer edges
- ✅ Higher precision

**File**: `state_core/graph/graph_builder.py`  
**Test**: Check degree distribution (should be less skewed)

---

### 1.3 K-1 Sampling (Not Every Step)
**Priority**: 🟢 LOW  
**Time**: 30 minutes  
**Complexity**: Very Low

**What**: Run K-1 attribution every 10 steps instead of every step

**Code**:
```python
# In training loop
if step % 10 == 0:
    attribution = k1_adapter.apply_hierarchical_updates(loss, state, step)
else:
    attribution = None
```

**Benefits**:
- ✅ 5-10% training speedup
- ✅ Still get interpretability signal

**File**: `test_sosm.py`, training loop  
**Test**: Verify still converges

---

### 1.4 Reduce State Update Layers
**Priority**: 🟡 MEDIUM  
**Time**: 1 hour (+ ablation)  
**Complexity**: Very Low

**What**: Reduce from 6 layers to 4 layers

**Code**:
```python
# In config
'model': {
    'n_layers': 4,  # Was 6
    'hidden_dim': 896,  # Increase slightly (was 768)
}
```

**Benefits**:
- ✅ 33% fewer computations
- ✅ 25% memory savings

**File**: `test_sosm.py`, config dict  
**Test**: Ablate 3, 4, 5, 6 layers; measure perplexity

---

### 1.5 Mixed Precision Training
**Priority**: 🟢 LOW  
**Time**: 1 hour  
**Complexity**: Very Low

**What**: Use FP16 for matmuls, FP32 for LayerNorm

**Code**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        logits, state = model(input_ids)
        loss = criterion(logits, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- ✅ 2× training speed
- ✅ 50% memory reduction

**File**: `test_sosm.py`, training loop  
**Test**: Verify no NaN gradients

---

## Phase 1 Metrics

**Expected Results**:
- Training speed: 1.0× → 1.45×
- Memory usage: 100% → 70%
- Perplexity: ≤ +0.5% (minimal degradation)
- Time investment: 1.5-2 days

**Validation**:
```bash
# Benchmark before
python benchmark.py --config baseline

# Implement Phase 1
# ...

# Benchmark after
python benchmark.py --config phase1 --compare baseline
```

---

# Phase 2: Quality & Interpretability 📈

**Timeline**: Week 2 (5-7 days)  
**Goal**: +8-10% disambiguation accuracy, interpretability  
**Risk Level**: 🟡 Low

## Items

### 2.1 Blockwise Similarity
**Priority**: 🔴 CRITICAL  
**Time**: 2-3 hours  
**Complexity**: Low

**What**: Compute similarity from weighted subset of MU blocks (I, R2, K)

**Code**:
```python
class BlockwiseSimilarity:
    def __init__(self):
        self.blocks = {
            'I': slice(0, 4),
            'R2': slice(12, 16),
            'K': slice(20, 24),
        }
        self.weights = {'I': 0.5, 'R2': 0.3, 'K': 0.2}
    
    def compute(self, semantic_state):
        sims = 0
        for name, weight in self.weights.items():
            block = semantic_state[:, self.blocks[name]]
            block_norm = F.normalize(block, dim=-1)
            sims += weight * (block_norm @ block_norm.T)
        return sims
```

**Benefits**:
- ✅ 2-3× speedup (24D vs 64D)
- ✅ Interpretable
- ✅ Less noise

**File**: `state_core/graph/graph_builder.py`  
**Test**: Ablate different block combinations

---

### 2.2 Adaptive K (Entropy-Based)
**Priority**: 🔴 CRITICAL  
**Time**: 4-6 hours  
**Complexity**: Medium

**What**: Ambiguous tokens get more edges; clear tokens get fewer

**Code**:
```python
def compute_adaptive_k(semantic_state, K_base=5, K_min=3, K_max=12, alpha=1.5):
    T = semantic_state.size(0)
    blocks = semantic_state.view(T, 16, 4)
    
    entropies = []
    for i in range(T):
        block_norms = blocks[i].norm(dim=-1)
        p = F.softmax(block_norms, dim=0)
        entropy = -(p * torch.log(p + 1e-8)).sum()
        entropies.append(entropy.item())
    
    # Normalize and scale
    ent_tensor = torch.tensor(entropies)
    ent_norm = (ent_tensor - ent_tensor.mean()) / (ent_tensor.std() + 1e-6)
    
    K_values = []
    for ent in ent_norm:
        K_i = K_base + alpha * ent.item()
        K_values.append(max(K_min, min(K_max, int(K_i))))
    
    return K_values
```

**Benefits**:
- ✅ "bank" (ambiguous) → K=10
- ✅ "the" (clear) → K=3
- ✅ 15-20% speedup

**File**: `state_core/graph/graph_builder.py`  
**Test**: Verify K increases with entropy

---

### 2.3 Edge Typing & Provenance
**Priority**: 🟡 HIGH  
**Time**: 6-8 hours  
**Complexity**: Medium

**What**: Store WHY each edge exists (block contributions)

**Code**:
```python
class EdgeWithProvenance:
    def __init__(self, src, dst, block_sims):
        self.src = src
        self.dst = dst
        self.block_sims = block_sims  # {'I': 0.3, 'R2': 0.2, 'K': 0.4}
        self.dominant = max(block_sims.items(), key=lambda x: x[1])[0]
        self.total_sim = sum(block_sims.values())

def build_with_provenance(semantic_state, edges):
    edges_prov = []
    for i, j in edges:
        block_sims = {}
        for name, slice_obj in blocks.items():
            sim = F.cosine_similarity(
                semantic_state[i, slice_obj],
                semantic_state[j, slice_obj],
                dim=0
            )
            block_sims[name] = sim.item()
        
        edges_prov.append(EdgeWithProvenance(i, j, block_sims))
    return edges_prov
```

**Benefits**:
- ✅ Explains edge creation
- ✅ Enables block ablations
- ✅ Research insights

**File**: `state_core/graph/graph_builder.py`  
**Test**: Log block distributions

---

### 2.4 Low-Dim Similarity Projection
**Priority**: 🟡 MEDIUM  
**Time**: 3-4 hours  
**Complexity**: Low-Medium

**What**: Project 64D → 16D for similarity computation only

**Code**:
```python
class SimilarityProjector:
    def __init__(self, d_in=64, d_proj=16):
        # Fixed random orthonormal projection
        P = torch.randn(d_in, d_proj)
        P, _ = torch.qr(P)
        self.register_buffer('P', P)  # NOT learned
    
    def compute_sim(self, semantic_state):
        proj = semantic_state @ self.P  # [T, 16]
        proj_norm = F.normalize(proj, dim=-1)
        return proj_norm @ proj_norm.T
```

**Benefits**:
- ✅ 3-4× faster similarity
- ✅ Deterministic
- ✅ Minimal quality loss

**File**: `state_core/graph/graph_builder.py`  
**Test**: Compare with full 64D

---

### 2.5 Forward Attribution
**Priority**: 🟢 LOW  
**Time**: 4-5 hours  
**Complexity**: Medium

**What**: Track which MU blocks contribute to predictions

**Code**:
```python
def forward_attribution(model, input_ids):
    activations = {}
    
    def hook(name):
        def fn(module, input, output):
            activations[name] = output.detach()
        return fn
    
    for name, module in model.named_modules():
        if 'mu' in name or 'block' in name:
            module.register_forward_hook(hook(name))
    
    logits = model(input_ids)
    
    # Analyze block contributions
    return analyze_contributions(activations)
```

**Benefits**:
- ✅ Real-time interpretability
- ✅ Complements K-1

**File**: `state_core/analysis/attribution.py` (new)  
**Test**: Verify contributions sum to 1.0

---

## Phase 2 Metrics

**Expected Results**:
- Disambiguation accuracy: +8-10%
- Graph construction: 3× faster
- Interpretability: High (block provenance + forward attribution)
- Perplexity: -2% (improvement)

---

# Phase 3: Scale & Advanced Features 🚀

**Timeline**: Week 3-4 (10-14 days)  
**Goal**: 2× total speed, 50% memory, handle T > 512  
**Risk Level**: 🟡 Medium

## Items

### 3.1 TEMPORAL-Gated Edge Acceptance
**Priority**: 🟡 MEDIUM  
**Time**: 8-10 hours  
**Complexity**: High

**What**: Use TEMPORAL to gate edges without mixing in similarity

**Code**:
```python
class TemporalGate(nn.Module):
    def __init__(self, temporal_dim=32):
        super().__init__()
        self.proj = nn.Linear(temporal_dim, temporal_dim)
    
    def forward(self, t_i, t_j):
        proj_i = self.proj(t_i)
        proj_j = self.proj(t_j)
        return torch.sigmoid(torch.dot(proj_i, proj_j))

def filter_with_temporal(edges, semantic_state, temporal_state,
                         tau_mu=0.15, tau_temp=0.3, tau_high=0.6):
    gate = TemporalGate()
    filtered = []
    
    for i, j in edges:
        sem_sim = cosine_sim(semantic_state[i], semantic_state[j])
        temp_score = gate(temporal_state[i], temporal_state[j])
        
        accept = (sem_sim > tau_mu and temp_score > tau_temp) or (sem_sim > tau_high)
        
        if accept:
            filtered.append((i, j, {'sem': sem_sim, 'temp': temp_score}))
    
    return filtered
```

**Benefits**:
- ✅ Filters implausible co-occurrences
- ✅ +3-5% accuracy
- ✅ Doesn't contaminate similarity

**File**: `state_core/graph/graph_builder.py`  
**Test**: Ablate thresholds

---

### 3.2 Deterministic Shortcuts
**Priority**: 🟡 MEDIUM  
**Time**: 3-4 hours  
**Complexity**: Medium

**What**: Replace random shortcuts with bridging-based

**Code**:
```python
def deterministic_shortcuts(graph, ratio=0.20):
    import networkx as nx
    
    G = nx.Graph()
    G.add_edges_from(graph['edges'])
    
    # Betweenness centrality (bridging score)
    centrality = nx.betweenness_centrality(G)
    
    # Top bridges
    sorted_nodes = sorted(centrality.items(), key=lambda x: -x[1])
    n_bridges = int(ratio * len(sorted_nodes))
    bridges = [node for node, _ in sorted_nodes[:n_bridges]]
    
    # Connect bridges
    shortcuts = []
    for i, b1 in enumerate(bridges):
        for b2 in bridges[i+1:]:
            if not G.has_edge(b1, b2):
                shortcuts.append((b1, b2))
    
    return shortcuts
```

**Benefits**:
- ✅ Deterministic (reproducible)
- ✅ Explainable
- ✅ Better structure

**File**: `state_core/graph/graph_builder.py`  
**Test**: Verify reproducibility

---

### 3.3 Sparse Attention Kernels
**Priority**: 🔴 CRITICAL  
**Time**: 16-20 hours  
**Complexity**: Very High

**What**: Use sparse attention that only computes for edges

**Code** (requires FlashAttention or custom):
```python
# Pseudo-code (complex implementation)
from flash_attn import flash_attn_func

def sparse_attention(Q, K, V, edge_index, edge_weight):
    # This requires FlashAttention v2 with mask support
    # Or custom CUDA kernel
    
    # Convert edge_index to mask
    mask = torch.zeros(T, T, dtype=torch.bool)
    mask[edge_index[0], edge_index[1]] = True
    
    # Flash attention with mask
    output = flash_attn_func(Q, K, V, causal=False, custom_mask=mask)
    return output
```

**Benefits**:
- ✅ 40% memory reduction
- ✅ 20-30% speed for sparse graphs
- ✅ Scales to T > 512

**File**: `state_core/pipeline.py`  
**Test**: Benchmark vs dense attention

**Note**: This is the most complex item - consider skipping if short on time

---

### 3.4 Graph Caching
**Priority**: 🟢 LOW  
**Time**: 2-3 hours  
**Complexity**: Low-Medium

**What**: Cache graph based on token IDs (inference only)

**Code**:
```python
class GraphCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_or_build(self, token_ids, build_fn):
        key = tuple(token_ids.tolist())
        
        if key in self.cache:
            return self.cache[key]
        
        graph = build_fn(token_ids)
        
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = graph
        return graph
```

**Benefits**:
- ✅ 50%+ inference speedup
- ✅ Critical for generation

**File**: `state_core/graph/graph_cache.py` (new)  
**Test**: Measure hit rate

---

### 3.5 Factorized Embeddings (ALBERT-style)
**Priority**: 🟡 MEDIUM  
**Time**: 2-3 hours  
**Complexity**: Low

**What**: Reduce embedding parameters via factorization

**Code**:
```python
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, factorized_dim=128):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, factorized_dim)
        self.projection = nn.Linear(factorized_dim, embed_dim)
    
    def forward(self, token_ids):
        factorized = self.word_embeddings(token_ids)  # [B, T, 128]
        projected = self.projection(factorized)  # [B, T, 768]
        return projected
```

**Benefits**:
- ✅ 6× parameter reduction (38M → 6.6M)
- ✅ No semantic change
- ✅ Easy to implement

**File**: `state_core/adapters/mu_adapter.py`  
**Test**: Verify no quality loss

---

## Phase 3 Metrics

**Expected Results**:
- Training speed: 1.45× → 2.0×
- Memory: 70% → 50%
- Max sequence: 128 → 512+
- Perplexity: -4% (total improvement)

---

# Phase 4: Long-Range & Efficiency 🌟

**Timeline**: Week 5-6 (10-14 days)  
**Goal**: Infinite context, parameter efficiency, production polish  
**Risk Level**: 🟡 Medium

## Items

### 4.1 HNSW Memory (Infinite Context)
**Priority**: 🔴 CRITICAL  
**Time**: 8-12 hours  
**Complexity**: High

**What**: Hierarchical navigable small-world index for long-term memory

**Code**:
```python
import hnswlib

class HNSWMemory:
    def __init__(self, dim=64, max_elements=1000000):
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )
        self.states = []  # Store full states
    
    def add(self, semantic_state, metadata):
        """Add state to long-term memory."""
        idx = len(self.states)
        self.index.add_items(semantic_state.cpu().numpy(), [idx])
        self.states.append({
            'state': semantic_state,
            'metadata': metadata
        })
    
    def retrieve(self, query_state, k=20):
        """Retrieve K most similar past states."""
        labels, distances = self.index.knn_query(
            query_state.cpu().numpy(), k=k
        )
        
        retrieved = []
        for idx, dist in zip(labels[0], distances[0]):
            retrieved.append({
                'state': self.states[idx]['state'],
                'similarity': 1 - dist,  # Convert distance to similarity
                'metadata': self.states[idx]['metadata']
            })
        
        return retrieved
    
    def integrate_into_graph(self, current_state, graph, k=10):
        """Add long-range edges from retrieved memories."""
        retrieved = self.retrieve(current_state, k=k)
        
        # Add edges to retrieved states
        long_range_edges = []
        for item in retrieved:
            if item['similarity'] > 0.4:
                long_range_edges.append({
                    'type': 'memory',
                    'similarity': item['similarity'],
                    'metadata': item['metadata']
                })
        
        graph['memory_edges'] = long_range_edges
        return graph
```

**Benefits**:
- ✅ O(log N) retrieval (not O(N))
- ✅ Infinite context window
- ✅ Cross-document reasoning

**File**: `state_core/memory/hnsw_memory.py` (new)  
**Test**: Retrieve accuracy on held-out states

---

### 4.2 LoRA Adapters (Efficient Fine-Tuning)
**Priority**: 🟡 MEDIUM  
**Time**: 6-8 hours  
**Complexity**: Medium

**What**: Low-rank adaptation for domain specialization

**Code**:
```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad = False  # Freeze base
        
        # LoRA decomposition
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
    
    def forward(self, x):
        base_output = self.base(x)
        lora_output = (x @ self.lora_A) @ self.lora_B
        return base_output + self.scaling * lora_output

def apply_lora(model, rank=8):
    """Convert all Linear layers to LoRA."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Replace with LoRA
            in_feat = module.in_features
            out_feat = module.out_features
            lora_module = LoRALinear(in_feat, out_feat, rank=rank)
            lora_module.base.weight.data = module.weight.data.clone()
            lora_module.base.bias.data = module.bias.data.clone() if module.bias is not None else None
            
            # Replace in parent
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, lora_module)
```

**Benefits**:
- ✅ 100× fewer trainable params
- ✅ Modular domain adaptation
- ✅ Swap adapters easily

**File**: `state_core/adapters/lora.py` (new)  
**Test**: Fine-tune on small domain, measure accuracy

---

### 4.3 KV Caching (Inference Speedup)
**Priority**: 🟡 HIGH  
**Time**: 4-6 hours  
**Complexity**: Medium

**What**: Cache previous tokens' K/V during generation

**Code**:
```python
class KVCache:
    def __init__(self):
        self.past_keys = []
        self.past_values = []
    
    def update(self, new_keys, new_values):
        """Append new K/V to cache."""
        if len(self.past_keys) == 0:
            self.past_keys = [new_keys]
            self.past_values = [new_values]
        else:
            # Concatenate along sequence dimension
            self.past_keys.append(new_keys)
            self.past_values.append(new_values)
        
        return (
            torch.cat(self.past_keys, dim=1),
            torch.cat(self.past_values, dim=1)
        )
    
    def clear(self):
        self.past_keys = []
        self.past_values = []

# In generation loop
kv_cache = KVCache()

for step in range(max_new_tokens):
    # Only compute Q for new token, use cached K/V
    Q_new = compute_Q(new_token)
    K_new, V_new = compute_KV(new_token)
    
    # Update cache
    K_all, V_all = kv_cache.update(K_new, V_new)
    
    # Attention with full K/V
    output = attention(Q_new, K_all, V_all)
```

**Benefits**:
- ✅ 3-5× inference speedup
- ✅ Critical for generation

**File**: `state_core/pipeline.py`  
**Test**: Generate 100 tokens, measure speed

---

### 4.4 Gradient Checkpointing
**Priority**: 🟢 LOW  
**Time**: 2-3 hours  
**Complexity**: Low

**What**: Trade compute for memory savings

**Code**:
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x, *args):
        if self.training:
            return checkpoint(self.layer, x, *args)
        else:
            return self.layer(x, *args)

# Wrap expensive layers
for i, layer in enumerate(model.operators):
    if i > 0:  # Checkpoint all but first layer
        model.operators[i] = CheckpointedLayer(layer)
```

**Benefits**:
- ✅ 40-50% memory reduction
- ✅ 15-20% slower (acceptable tradeoff)

**File**: `state_core/pipeline.py`  
**Test**: Measure memory vs speed tradeoff

---

### 4.5 Multi-Block Attention Heads
**Priority**: 🟢 LOW  
**Time**: 6-8 hours  
**Complexity**: High

**What**: Specialized attention heads per MU block type

**Code**:
```python
class BlockSpecializedAttention(nn.Module):
    def __init__(self, dim, n_heads, n_blocks=16):
        super().__init__()
        
        # Each head specializes in one block
        self.block_heads = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads=1)
            for _ in range(n_blocks)
        ])
        
        # Global cross-block heads
        self.global_head = nn.MultiheadAttention(dim, num_heads=4)
    
    def forward(self, x, block_masks):
        # x: [B, T, 768]
        # block_masks: [16, T, T] - one mask per block type
        
        outputs = []
        
        # Block-specific attention
        for i, head in enumerate(self.block_heads):
            out, _ = head(x, x, x, attn_mask=block_masks[i])
            outputs.append(out)
        
        # Global attention
        global_out, _ = self.global_head(x, x, x)
        outputs.append(global_out)
        
        # Combine
        return sum(outputs) / len(outputs)
```

**Benefits**:
- ✅ Interpretable (know which blocks matter)
- ✅ Specialized processing

**File**: `state_core/pipeline.py`  
**Test**: Compare with standard attention

---

## Phase 4 Metrics

**Expected Results**:
- Context window: 512 → Infinite (via HNSW)
- Embedding params: 38M → 6.6M
- Inference speed: +3-5× (KV cache)
- Memory: 50% → 35% (checkpointing)

---

# Phase 5: Production & Deployment 🏭

**Timeline**: Week 7-8 (10-14 days)  
**Goal**: Production-ready, multi-GPU, serving infrastructure  
**Risk Level**: 🟢 Low

## Items

### 5.1 Multi-GPU Training (DataParallel)
**Priority**: 🔴 CRITICAL  
**Time**: 6-8 hours  
**Complexity**: Medium

**Code**:
```python
import torch.nn as nn
from torch.nn.parallel import DataParallel

# Wrap model
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = DataParallel(model)

# Training loop unchanged
for batch in train_loader:
    # Automatically splits batch across GPUs
    logits, state = model(input_ids)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

**Benefits**:
- ✅ Linear speedup with # GPUs
- ✅ Handle larger batches

**File**: `test_sosm.py`  
**Test**: 2 GPU vs 4 GPU speedup

---

### 5.2 Model Quantization (INT8)
**Priority**: 🟡 MEDIUM  
**Time**: 4-6 hours  
**Complexity**: Medium

**Code**:
```python
import torch.quantization

# Quantize model
model_fp32 = copy.deepcopy(model)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear},
    dtype=torch.qint8
)

# Inference
with torch.no_grad():
    output = model_int8(input_ids)
```

**Benefits**:
- ✅ 4× smaller model
- ✅ 2-3× faster inference
- ✅ -1% accuracy (acceptable)

**File**: `deployment/quantize.py` (new)  
**Test**: Measure accuracy drop

---

### 5.3 ONNX Export
**Priority**: 🟡 MEDIUM  
**Time**: 4-6 hours  
**Complexity**: Medium

**Code**:
```python
import torch.onnx

# Export to ONNX
dummy_input = torch.randint(0, 50257, (1, 64))
torch.onnx.export(
    model,
    dummy_input,
    "sosm.onnx",
    export_params=True,
    opset_version=14,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch', 1: 'sequence'}
    }
)

# Load in ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("sosm.onnx")
```

**Benefits**:
- ✅ Deploy anywhere (C++, JS, mobile)
- ✅ Optimized inference

**File**: `deployment/export_onnx.py` (new)  
**Test**: Verify outputs match PyTorch

---

### 5.4 REST API Server
**Priority**: 🟡 MEDIUM  
**Time**: 8-10 hours  
**Complexity**: Medium

**Code**:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str
    max_length: int = 50

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Tokenize
    tokens = tokenizer.encode(request.text)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            torch.tensor([tokens]),
            max_length=request.max_length
        )
    
    # Decode
    text = tokenizer.decode(output[0])
    
    return {"generated_text": text}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

**Benefits**:
- ✅ HTTP API for inference
- ✅ Easy integration

**File**: `deployment/server.py` (new)  
**Test**: Load test with locust

---

### 5.5 Model Distillation
**Priority**: 🟢 LOW  
**Time**: 12-16 hours  
**Complexity**: High

**Code**:
```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard loss (student vs labels)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss (student vs teacher)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction='batchmean'
        ) * (self.T ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Train smaller student model
student = create_small_sosm(n_layers=2, hidden_dim=384)
teacher = load_pretrained_sosm()

for batch in train_loader:
    with torch.no_grad():
        teacher_logits = teacher(input_ids)
    
    student_logits = student(input_ids)
    loss = distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()
```

**Benefits**:
- ✅ 5-10× smaller model
- ✅ 3-5× faster inference
- ✅ 90-95% of teacher accuracy

**File**: `training/distillation.py` (new)  
**Test**: Compare student vs teacher accuracy

---

### 5.6 Monitoring & Logging
**Priority**: 🟡 MEDIUM  
**Time**: 4-6 hours  
**Complexity**: Low

**Code**:
```python
import wandb
from prometheus_client import Counter, Histogram

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
latency_histogram = Histogram('prediction_latency_seconds', 'Prediction latency')

# W&B logging
wandb.init(project="sosm")

for step, batch in enumerate(train_loader):
    logits, state = model(input_ids)
    loss = criterion(logits, labels)
    
    # Log to W&B
    wandb.log({
        'loss': loss.item(),
        'perplexity': torch.exp(loss).item(),
        'num_edges': state.routing_state['num_edges'],
        'semantic_edges': state.routing_state['edge_types']['semantic'],
        'avg_degree': avg_degree,
    })
```

**Benefits**:
- ✅ Real-time monitoring
- ✅ Debugging support

**File**: `utils/monitoring.py` (new)  
**Test**: Set up dashboard

---

## Phase 5 Metrics

**Expected Results**:
- Multi-GPU: 4× training speedup
- Quantization: 4× smaller, 2× faster
- API: 100 req/sec throughput
- Distilled model: 10× smaller, 95% accuracy

---

# Summary Timeline

| Week | Phase                        | Key Deliverables                                    |
|------|------------------------------|-----------------------------------------------------|
| 1    | Phase 1: Quick Wins          | Streaming Top-K, Mutual k-NN, Mixed precision       |
| 2    | Phase 2: Quality             | Blockwise sim, Adaptive K, Edge provenance          |
| 3-4  | Phase 3: Scale               | TEMPORAL gate, Sparse attention, Factorized embed   |
| 5-6  | Phase 4: Long-Range          | HNSW memory, LoRA, KV cache                         |
| 7-8  | Phase 5: Production          | Multi-GPU, Quantization, API, Monitoring            |

---

# Risk Mitigation

**For each phase**:
1. ✅ Implement items sequentially
2. ✅ Test after each item
3. ✅ Benchmark before moving to next
4. ✅ Maintain baseline comparison
5. ✅ Document any regressions

**Rollback plan**:
- All changes behind config flags
- Can revert to baseline anytime
- Git branches per phase

---

# Success Criteria

**Phase 1**: 45% faster, 30% less memory, perplexity ≤ +0.5%  
**Phase 2**: +8% accuracy, 3× faster graph building  
**Phase 3**: 2× total speed, T=512 support  
**Phase 4**: Infinite context, 6× fewer params  
**Phase 5**: Multi-GPU, production API, monitoring

---

**Ready to start Phase 1?** 🚀

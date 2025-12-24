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

# Phase 6: Advanced Architecture - Mamba, RoPE & Graphormer Fusion 🔬

**Timeline**: Week 9-12 (3-4 weeks)  
**Goal**: Harmonize Graph Transformers, SSMs, and positional embeddings for high-efficiency token flow  
**Risk Level**: 🔴 Very High (Research-level complexity)

> **WARNING**: Phase 6 represents a fundamental architectural redesign. This should only be pursued after Phases 1-5 are complete and validated. Consider this a separate research project.

---

## 📋 Executive Analysis

The Sparsely Organized Semantic Model (SOSM) represents a sophisticated attempt to unify three divergent paradigms:

1. **Graphormer** - Structural inductive biases via graph topology
2. **Mamba (SSM)** - Linear-time sequence modeling  
3. **RoPE** - Relative positional semantics

### The Tri-Partite Constraint Dilemma

**Problem**: Incompatibility between:
- Graphormer's O(N³) shortest path distance (SPD) computation
- Mamba's O(N) linear recurrence requirement
- RoPE's 1D assumptions vs. graph topology

**Solution**: This phase provides a re-architected framework that resolves these conflicts.

---

## 6.1 Structural Encoding: Breaking the Cubic Bottleneck

**Priority**: 🔴 CRITICAL  
**Time**: 2-3 weeks  
**Complexity**: Very High

### Problem: Floyd-Warshall O(V³) Bottleneck

Current Graphormer implementations compute exact SPD between all node pairs:
- **Floyd-Warshall**: O(V³) complexity
- **All-Pairs BFS**: O(V·E) complexity
- **Scalability limit**: Cannot handle graphs > 5k nodes

### Solution: Landmark-Based Approximate SPD

**Concept**: Approximate global geometry using k landmarks instead of all-pairs distances.

**Code**:
```python
class LandmarkSPDEncoder:
    def __init__(self, num_landmarks=64):
        """
        Landmark-based approximate shortest path distance encoder.
        Complexity: O(k·E) instead of O(V³)
        """
        self.num_landmarks = num_landmarks
        self.distance_mlp = nn.Sequential(
            nn.Linear(num_landmarks * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def select_landmarks(self, graph):
        """
        Select landmarks based on degree centrality.
        High-degree hubs ensure good coverage.
        """
        degrees = graph.in_degrees() + graph.out_degrees()
        _, landmark_indices = torch.topk(degrees, self.num_landmarks)
        return landmark_indices
    
    def compute_landmark_distances(self, graph, landmarks):
        """
        Compute distances from each landmark to all nodes.
        k runs of BFS: O(k·E)
        """
        num_nodes = graph.number_of_nodes()
        landmark_distances = torch.zeros(self.num_landmarks, num_nodes)
        
        for i, landmark in enumerate(landmarks):
            # Single-source BFS from landmark
            distances = self._bfs_distances(graph, landmark)
            landmark_distances[i] = distances
        
        return landmark_distances.T  # [num_nodes, num_landmarks]
    
    def approximate_distance(self, landmark_dist_i, landmark_dist_j):
        """
        Approximate distance between nodes i and j using triangle inequality.
        
        For nodes u, v and landmarks L:
        d(u,v) ≈ learned_function(|d(u,L) - d(v,L)| for all L)
        """
        # Concatenate landmark distance vectors
        combined = torch.cat([landmark_dist_i, landmark_dist_j], dim=-1)
        
        # MLP learns to approximate SPD from landmark coordinates
        approx_dist = self.distance_mlp(combined)
        return approx_dist
```

**Benefits**:
- ✅ Complexity: O(V³) → O(k·E) where k << V
- ✅ Dynamic graphs: Incremental updates possible
- ✅ Scalability: Handles 100k+ node graphs

**Alternative: Linear-Time Encodings**

For extreme efficiency:
```python
class LinearTimePE:
    """
    Spiking Node Tokenization (GT-SNT) or Random Walk PE.
    O(E) complexity.
    """
    def compute_rwpe(self, graph, walk_length=16):
        """Random walk positional encoding."""
        # Start random walks from each node
        walks = self._random_walks(graph, walk_length)
        
        # Encode walk statistics as positional features
        pe = self._encode_walk_statistics(walks)
        return pe
```

---

## 6.2 Differential & Sparse Attention

**Priority**: 🔴 CRITICAL  
**Time**: 2-3 weeks  
**Complexity**: Very High

### 6.2.1 Differential Transformer

**Problem**: Standard attention "smears" probability mass, causing noise sensitivity.

**Solution**: Subtract noise-capturing attention map from signal map.

**Code**:
```python
class DifferentialAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Two separate attention maps
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj_signal = nn.Linear(dim, dim)
        self.k_proj_noise = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Learnable noise suppression weight
        self.lambda_param = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x, graph_bias=None):
        B, T, C = x.shape
        
        Q = self.q_proj(x).view(B, T, self.num_heads, -1)
        K_signal = self.k_proj_signal(x).view(B, T, self.num_heads, -1)
        K_noise = self.k_proj_noise(x).view(B, T, self.num_heads, -1)
        V = self.v_proj(x).view(B, T, self.num_heads, -1)
        
        # Signal attention map
        attn_signal = (Q @ K_signal.transpose(-2, -1)) * self.scale
        if graph_bias is not None:
            attn_signal = attn_signal + graph_bias
        attn_signal = F.softmax(attn_signal, dim=-1)
        
        # Noise attention map
        attn_noise = (Q @ K_noise.transpose(-2, -1)) * self.scale
        attn_noise = F.softmax(attn_noise, dim=-1)
        
        # Differential attention: signal - λ*noise
        attn_diff = attn_signal - self.lambda_param * attn_noise
        
        # Apply to values
        out = attn_diff @ V
        return out.reshape(B, T, C)
```

**Benefits**:
- ✅ Sparser attention patterns
- ✅ Better noise filtering
- ✅ Enhances semantic focus

### 6.2.2 Approximate Nearest Neighbor Attention (ANNA)

**For long contexts where N² is prohibitive**:

**Code**:
```python
class ANNAttention(nn.Module):
    def __init__(self, dim, num_buckets=64):
        super().__init__()
        self.num_buckets = num_buckets
        
        # LSH hash functions
        self.hash_projections = nn.Parameter(
            torch.randn(dim, num_buckets)
        )
    
    def lsh_hash(self, x):
        """
        Locality-Sensitive Hashing for semantic + structural grouping.
        """
        # Project to hash space
        projections = x @ self.hash_projections
        
        # Binary hash codes
        hashes = (projections > 0).long()
        return hashes
    
    def forward(self, Q, K, V, landmark_encoding=None):
        """
        Attention with ANN retrieval.
        Complexity: O(N log N) instead of O(N²)
        """
        # Hash queries and keys
        if landmark_encoding is not None:
            # Include structural information in hash
            Q_aug = torch.cat([Q, landmark_encoding], dim=-1)
            K_aug = torch.cat([K, landmark_encoding], dim=-1)
        else:
            Q_aug, K_aug = Q, K
        
        q_hashes = self.lsh_hash(Q_aug)
        k_hashes = self.lsh_hash(K_aug)
        
        # Group tokens by hash buckets
        # Only compute attention within buckets
        output = self._bucketed_attention(Q, K, V, q_hashes, k_hashes)
        return output
```

**Benefits**:
- ✅ O(N²) → O(N log N) complexity
- ✅ Structural LSH ensures topologically close tokens are retrieved

### 6.2.3 Block-Sparse Triton Kernel

**Hardware-optimized sparse attention**:

**Triton Code**:
```python
import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q, K, V, Out,
    landmark_dist,  # [N, k] landmark distances
    stride_qm, stride_kn, stride_vn,
    block_size: tl.constexpr,
    threshold: tl.constexpr
):
    """
    Block-sparse attention with topological masking.
    Only computes blocks where tokens are structurally connected.
    """
    # Block indices
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    
    # Load landmark metadata into SRAM
    offs_m = block_m * block_size + tl.arange(0, block_size)
    offs_n = block_n * block_size + tl.arange(0, block_size)
    
    landmark_m = tl.load(landmark_dist + offs_m * stride_qm)
    landmark_n = tl.load(landmark_dist + offs_n * stride_qm)
    
    # Compute approximate structural distance
    dist = tl.abs(landmark_m[:, None] - landmark_n[None, :])
    dist_score = tl.sum(dist, axis=-1)  # L1 distance in landmark space
    
    # Early exit if block is irrelevant
    if tl.min(dist_score) > threshold:
        return  # Don't load K, V for this block
    
    # Load Q, K, V blocks (only if structurally relevant)
    q_block = tl.load(Q + offs_m[:, None] * stride_qm)
    k_block = tl.load(K + offs_n[:, None] * stride_kn)
    v_block = tl.load(V + offs_n[:, None] * stride_vn)
    
    # Compute attention scores
    scores = tl.dot(q_block, tl.trans(k_block))
    scores = scores / tl.sqrt(tl.float32(q_block.shape[-1]))
    
    # Softmax
    scores = tl.softmax(scores, axis=-1)
    
    # Apply to values
    out_block = tl.dot(scores, v_block)
    
    # Store output
    tl.store(Out + offs_m[:, None] * stride_qm, out_block)
```

**Benefits**:
- ✅ 40% memory reduction
- ✅ SRAM-resident distance checks (no HBM stalls)
- ✅ Gradient flow via Straight-Through Estimator

---

## 6.3 Graph-Mamba Hybrid Architecture

**Priority**: 🔴 CRITICAL  
**Time**: 2-3 weeks  
**Complexity**: Very High

### The Challenge

Mamba's recurrence (h_t = A·h_{t-1} + B·x_t) cannot naturally incorporate random-access graph structure.

### Solution: Hybrid Mamba-Attention

**Architecture**:
- 80% Mamba layers (global context mixing, O(N))
- 20% Sparse Graph Attention layers (structural refinement, O(N log N))

**Code**:
```python
class GraphMambaHybrid(nn.Module):
    def __init__(self, dim=768, num_layers=24, mamba_layers_per_attn=4):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i % mamba_layers_per_attn == 0:
                # Sparse Graph Attention layer
                self.layers.append(
                    LandmarkSparseAttention(dim, num_landmarks=64)
                )
            else:
                # Topologically-Adaptive Mamba layer
                self.layers.append(
                    TopologicalMambaBlock(dim)
                )
    
    def forward(self, x, graph_encoding):
        for layer in self.layers:
            if isinstance(layer, TopologicalMambaBlock):
                x = layer(x, graph_encoding)
            else:
                x = layer(x, graph_encoding)
        return x

class TopologicalMambaBlock(nn.Module):
    """
    Mamba with graph-adaptive discretization.
    """
    def __init__(self, dim):
        super().__init__()
        self.ssm = SelectiveSSM(dim)
        
        # Graph encoding modulates Δ (discretization step)
        self.delta_proj = nn.Linear(dim + 64, dim)  # +64 for landmark encoding
    
    def forward(self, x, landmark_encoding):
        """
        If current token is topologically far from previous:
          → Large Δ (reset state)
        If close:
          → Small Δ (preserve state)
        """
        # Combine token features with structural position
        combined = torch.cat([x, landmark_encoding], dim=-1)
        
        # Compute topology-adaptive Δ
        delta = F.softplus(self.delta_proj(combined))
        
        # Run SSM with adaptive discretization
        output = self.ssm(x, delta=delta)
        return output
```

**Benefits**:
- ✅ O(N) global mixing (Mamba)
- ✅ O(N log N) structural refinement (Sparse Attention)
- ✅ Best of both worlds

---

## 6.4 Graph-RoPE: Generalized Positional Encoding

**Priority**: 🟡 HIGH  
**Time**: 1-2 weeks  
**Complexity**: High

### Problem

Standard RoPE assumes 1D sequence (position m-n is scalar). Graphs have non-Euclidean topology.

### Solution: Landmark-Relative RoPE

**Concept**: Use landmark distance vectors as multi-dimensional "positions".

**Code**:
```python
class GraphRoPE(nn.Module):
    def __init__(self, dim, num_landmarks=64):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.dim = dim
        
        # Each landmark defines a rotation subspace
        self.freqs_per_landmark = self._init_frequencies(dim // num_landmarks)
    
    def _init_frequencies(self, subdim):
        """Initialize rotation frequencies."""
        freqs = 1.0 / (10000 ** (torch.arange(0, subdim, 2).float() / subdim))
        return freqs
    
    def forward(self, x, landmark_distances):
        """
        x: [B, T, dim]
        landmark_distances: [B, T, num_landmarks]
        
        Apply rotation based on structural position (landmark coordinates).
        """
        B, T, D = x.shape
        
        # Reshape for per-landmark rotation
        x = x.view(B, T, self.num_landmarks, -1)
        
        # For each landmark dimension
        rotated = []
        for i in range(self.num_landmarks):
            # Distance to this landmark
            pos = landmark_distances[:, :, i]  # [B, T]
            
            # Compute rotation angles
            freqs = self.freqs_per_landmark[i]
            angles = pos.unsqueeze(-1) * freqs  # [B, T, subdim//2]
            
            # Apply rotation to this subspace
            x_i = x[:, :, i, :]  # [B, T, subdim]
            x_rotated = self._apply_rotary_emb(x_i, angles)
            rotated.append(x_rotated)
        
        # Concatenate all rotated subspaces
        output = torch.cat(rotated, dim=-1)
        return output
    
    def _apply_rotary_emb(self, x, angles):
        """Apply 2D rotation matrices."""
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # Split into pairs
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Rotate
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        # Interleave back
        x_rot = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)
        return x_rot
```

**Alternative: ALiBi for Graphs**

```python
class GraphALiBi(nn.Module):
    """
    Attention with Linear Biases adapted for graphs.
    Simpler than RoPE, better extrapolation.
    """
    def __init__(self, num_heads):
        super().__init__()
        # Per-head slope
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
    
    def forward(self, attn_scores, landmark_distances_i, landmark_distances_j):
        """
        Add bias: -m * approximate_distance(i, j)
        """
        # Approximate SPD from landmark coordinates
        approx_dist = torch.abs(
            landmark_distances_i.unsqueeze(-2) - 
            landmark_distances_j.unsqueeze(-3)
        ).sum(dim=-1)  # L1 distance
        
        # Apply per-head slope
        bias = -self.slopes.view(1, -1, 1, 1) * approx_dist.unsqueeze(1)
        
        return attn_scores + bias
```

**Benefits**:
- ✅ Generalizes RoPE to graphs
- ✅ Multi-dimensional structural encoding
- ✅ Better length/size extrapolation (ALiBi)

---

## 6.5 Semantic Routing: Soft MoE & Expert Choice

**Priority**: 🟡 MEDIUM  
**Time**: 1 week  
**Complexity**: Medium

### Problem

Standard Top-K MoE has training instabilities and expert collapse.

### Solution: Soft MoE + Expert Choice

**Code**:
```python
class SoftMoE(nn.Module):
    """
    Differentiable MoE via weighted averaging.
    No discrete routing → stable gradients.
    """
    def __init__(self, dim, num_experts=8, num_slots=128):
        super().__init__()
        self.num_experts = num_experts
        self.num_slots = num_slots
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
            for _ in range(num_experts)
        ])
        
        # Router: tokens → slot weights
        self.router = nn.Linear(dim, num_slots)
    
    def forward(self, x):
        B, T, D = x.shape
        
        # Compute slot weights
        slot_weights = F.softmax(self.router(x), dim=-1)  # [B, T, num_slots]
        
        # Tokens → Slots (differentiable pooling)
        slots = torch.einsum('btd,bts->bsd', x, slot_weights)  # [B, num_slots, D]
        
        # Each expert processes all slots
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(slots)
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, E, S, D]
        
        # Average expert outputs
        combined_slots = expert_outputs.mean(dim=1)  # [B, S, D]
        
        # Slots → Tokens (differentiable dispatch)
        output = torch.einsum('bsd,bts->btd', combined_slots, slot_weights)
        
        return output

class ExpertChoiceRouting(nn.Module):
    """
    Experts choose tokens (not vice versa).
    Guarantees load balancing.
    """
    def __init__(self, dim, num_experts=8, capacity_per_expert=64):
        super().__init__()
        self.num_experts = num_experts
        self.capacity = capacity_per_expert
        
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim * 4)
            for _ in range(num_experts)
        ])
        
        self.router = nn.Linear(dim, num_experts)
    
    def forward(self, x):
        B, T, D = x.shape
        
        # Compute affinity scores
        affinity = self.router(x)  # [B, T, E]
        
        # Each expert selects top-K tokens
        outputs = []
        for e in range(self.num_experts):
            scores_e = affinity[:, :, e]  # [B, T]
            _, top_indices = torch.topk(scores_e, self.capacity, dim=1)
            
            # Expert processes its chosen tokens
            tokens_e = torch.gather(
                x, 1, 
                top_indices.unsqueeze(-1).expand(-1, -1, D)
            )
            expert_out = self.experts[e](tokens_e)
            outputs.append(expert_out)
        
        # Combine (weighted by affinity)
        # ...implementation details...
        
        return combined_output
```

**Benefits**:
- ✅ Stable training (no discrete routing)
- ✅ Perfect load balancing (Expert Choice)
- ✅ Semantic specialization emerges naturally

---

## 6.6 Factorized Projections & LoRA

**Priority**: 🟡 MEDIUM  
**Time**: 1 week  
**Complexity**: Low-Medium

**Reduce 60-70% of parameters**:

**Code**:
```python
class FactorizedLinear(nn.Module):
    """
    W ≈ A × B where A: [d, r], B: [r, d]
    50-75% parameter reduction.
    """
    def __init__(self, in_dim, out_dim, rank=128):
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=True)
    
    def forward(self, x):
        return self.B(self.A(x))

# Replace all linear layers
def factorize_model(model, rank=128):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_dim = module.in_features
            out_dim = module.out_features
            
            # Replace with factorized version
            factorized = FactorizedLinear(in_dim, out_dim, rank)
            
            # SVD initialization for better starting point
            U, S, V = torch.svd(module.weight)
            factorized.A.weight.data = (U[:, :rank] @ torch.diag(S[:rank])).T
            factorized.B.weight.data = V[:, :rank].T
            
            # Replace in parent
            parent = _get_parent(model, name)
            setattr(parent, name.split('.')[-1], factorized)
```

**LoRA for domain adaptation**:
```python
# Already covered in Phase 4, but crucial for Phase 6
# Use LoRA for task-specific semantic specialization
```

---

## 6.7 Interpretability: DePass Framework

**Priority**: 🟢 LOW  
**Time**: 1 week  
**Complexity**: Medium

**Verify that complex components work as intended**:

**Code**:
```python
class DePass:
    """
    Decomposed Forward Pass for attribution.
    """
    def __init__(self, model):
        self.model = model
        self.cached_activations = {}
    
    def decompose_forward(self, x, component_labels):
        """
        Propagate components through frozen network.
        
        x: [B, T, D] can be viewed as sum of components
        component_labels: which component each token belongs to
        """
        # Run normal forward to cache activations
        _ = self.model(x)
        
        # Now run decomposed forward
        component_outputs = {}
        
        for comp_id in component_labels.unique():
            # Mask to this component
            comp_mask = (component_labels == comp_id).float()
            x_comp = x * comp_mask.unsqueeze(-1)
            
            # Forward with FIXED attention patterns and activations
            output_comp = self._forward_with_fixed_patterns(x_comp)
            component_outputs[comp_id] = output_comp
        
        return component_outputs
    
    def _forward_with_fixed_patterns(self, x):
        """
        Use cached attention patterns and nonlinearity masks.
        Makes propagation linear → allows exact attribution.
        """
        # Implementation uses cached self.cached_activations
        # ...
        pass
```

**Usage**:
```python
# Verify Expert Choice routing
depass = DePass(model)
outputs = depass.decompose_forward(x, expert_assignments)

# Check if Expert 1 only affects "structure" tokens
structural_contribution = outputs['expert_1'][structural_token_indices]
```

---

## 6.8 Hardware Optimization Summary

**Triton kernel requirements**:

1. **Block-Sparse Attention**: SRAM-resident metadata checks
2. **Mamba Parallel Scan**: Coalesced memory access for state matrices
3. **Soft Masking**: Straight-Through Estimator for gradient flow
4. **Expert Choice**: Efficient gather/scatter for token selection

**Memory hierarchy**:
- Landmark encodings → Shared Memory
- Distance computations → Registers  
- K/V blocks → Coalesced HBM reads (only if relevant)

---

## Phase 6 Architecture Comparison

| Component | Original SOSM | Phase 6 Optimized |
|-----------|---------------|-------------------|
| **Graph Encoding** | Floyd-Warshall O(V³) | Landmark Approx O(k·E) |
| **Token Mixing** | Dense Attention O(N²) | Graph-Mamba Hybrid O(N) |
| **Positional Enc** | Standard RoPE (1D) | Graph-RoPE (multi-dim) |
| **Routing** | Fixed/Random | Soft MoE / Expert Choice |
| **Projections** | Dense matrices | Factorized + LoRA |
| **Attention Kernel** | Standard CUDA | Triton Block-Sparse |
| **Interpretability** | Black box | DePass attribution |

---

## Expected Results

**Efficiency**:
- Graph encoding: O(V³) → O(k·E) (100× faster for large graphs)
- Attention: O(N²) → O(N log N) (10× faster)
- Total throughput: 5-10× improvement

**Effectiveness**:
- Better long-range reasoning (Mamba global context)
- Better structural reasoning (Landmark SPD + Differential Attention)
- Better semantic specialization (Soft MoE)

**Scalability**:
- Graphs: 5k nodes → 100k+ nodes
- Sequences: 512 → 8k+ tokens
- Parameters: 50-75% reduction via factorization

---

## ⚠️ Critical Warnings

1. **Complexity**: This is research-level work, not production optimization
2. **Dependencies**: Requires Phases 1-5 complete
3. **Risk**: High chance of instability during integration
4. **Timeline**: 3-4 weeks minimum, likely 2-3 months in practice
5. **Validation**: Extensive ablations required for each component

---

## Implementation Strategy

### Week 1-2: Foundation
- Implement Landmark SPD encoder
- Implement TopologicalMamba block
- Basic Graph-Mamba hybrid (no attention yet)

### Week 3-4: Attention Mechanisms
- Differential Attention
- Block-Sparse Triton kernels
- ANNA (if time permits)

### Week 5-6: Positional & Routing
- Graph-RoPE implementation
- Soft MoE or Expert Choice
- Factorized layers

### Week 7-8: Integration & Validation
- End-to-end integration
- DePass verification
- Extensive ablation studies

### Week 9-12: Optimization & Tuning
- Triton kernel optimization
- Hyperparameter tuning
- Performance benchmarking

---

## Success Criteria

**Phase 6 is successful if**:
- ✅ Graph encoding < 1 sec for 100k nodes
- ✅ End-to-end throughput 5× faster than Phase 5
- ✅ Accuracy maintained or improved
- ✅ Can handle 10k+ token sequences
- ✅ All components validated via DePass

---

## Recommended Reading

1. Graphormer: https://arxiv.org/abs/2106.05234
2. Mamba: https://arxiv.org/abs/2312.00752
3. Differential Transformers: https://arxiv.org/abs/2410.05258
4. Landmark-based Graph Embeddings: https://arxiv.org/abs/2106.10174
5. Soft MoE: https://arxiv.org/abs/2308.00951
6. DePass: https://arxiv.org/abs/2404.11444

---

**Ready to start Phase 1?** 🚀

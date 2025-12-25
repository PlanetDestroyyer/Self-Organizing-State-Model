# SOSM Implementation Phases: Revised Roadmap 🗺️

**Updated**: December 25, 2024  
**Based On**: Three research reports + current testing results

---

## 🎯 **Strategic Overview**

### **Current Status** (Phase 2.1 Complete)
- ✅ **PPL**: 3.67 (69% improvement from 12.00!)
- ✅ **Bug Fixes**: Threshold, shortcuts, config parameters all fixed
- ✅ **Qualitative Tests**: 11/11 disambiguation passed
- ❌ **Factual Recall**: 10% (data limitation, not architecture)
- ❌ **Homonym Separation**: 0.007 (MU position-invariance by design)
- ⚠️ **Generation**: Repetitive (sampling issue, easy fix)

### **Research Findings**

**Three comprehensive research reports analyzed**:
1. **Peer Review**: Conservative, incremental validation
2. **Modernization**: SOTA techniques (RoPE, FlashAttention)
3. **Advanced Research**: 12-month Ph.D.-level overhaul

**Synthesis**: Adopt Tier 1 (6 weeks), cherry-pick from Tier 2, defer Tier 3

---

## 📅 **Revised Phase Timeline**

| Phase | Focus | Duration | Risk | Outcome |
|-------|-------|----------|------|---------|
| **Phase 2.2** | Modern Foundations | Week 1 | 🟢 Very Low | PPL ~3.3, stable training |
| **Phase 2.3** | SOSM Core Validation | Week 2 | 🟡 Low-Medium | Context test, interpretability |
| **Phase 2.4** | SOTA Integration | Week 3 | 🟡 Medium | 3× speed, 50% factual recall |
| **Phase 2.5** | Comprehensive Analysis | Week 4-5 | 🟢 Low | Ablations, profiling |
| **Phase 3** | Advanced Features (Optional) | Month 3-6 | 🔴 High | Research contributions |
| **Future** | Fundamental Research | Year 2+ | 🔴 Very High | Ph.D. scope |

---

# Phase 2.2: Modern Foundations ⚡

**Timeline**: Week 1 (5-7 days)  
**Goal**: Add proven SOTA techniques with minimal risk  
**Risk Level**: 🟢 VERY LOW  
**Expected**: PPL 3.3-3.5, better generation, more stable training

## Why This Phase?

Current model (PPL 3.67) is excellent but uses older techniques:
- Fixed positional embeddings (not flexible for varying lengths)
- Standard LayerNorm (less stable than Pre-LN)
- Greedy decoding (causes repetition)

**Solution**: Swap in modern, proven alternatives

---

## Items

### 2.2.1 RoPE Positional Encodings
**Priority**: 🔴 CRITICAL  
**Time**: 4-6 hours  
**Complexity**: Medium

**What**: Replace fixed learned positional embeddings with Rotary Position Embeddings

**Current**:
```python
self.position_embeddings = nn.Embedding(max_len, dim)
```

**New**:
```python
class RoPEEmbedding(nn.Module):
    def __init__(self, dim, max_len=8192):
        super().__init__()
        # Compute frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len):
        # Generate rotation matrices
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    # Split into even/odd dimensions
    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]
    
    # Apply rotation
    q_rot = torch.cat([
        q_even * cos - q_odd * sin,
        q_even * sin + q_odd * cos
    ], dim=-1)
    k_rot = torch.cat([
        k_even * cos - k_odd * sin,
        k_even * sin + k_odd * cos
    ], dim=-1)
    
    return q_rot, k_rot
```

**Benefits**:
- ✅ No maximum length limit (extrapolates to unseen lengths)
- ✅ Better length generalization
- ✅ Relative position encoding (vs. absolute)
- ✅ Zero parameters (vs. learned embeddings)

**Alternative**: ALiBi (even simpler, adds linear bias to attention)

**File**: `state_core/adapters/mu_adapter.py`, `state_core/pipeline.py`  
**Test**: Train 2 epochs, verify PPL similar or better

**Source**: RoFormer paper, used in LLaMA, GPT-NeoX

---

### 2.2.2 Pre-LayerNorm Transformer
**Priority**: 🔴 CRITICAL  
**Time**: 2-3 hours  
**Complexity**: Low

**What**: Move LayerNorm INSIDE residual blocks (before attention/FFN)

**Current** (Post-LN):
```python
x = x + self.attn(self.ln1(x))
x = x + self.ffn(self.ln2(x))
```

**New** (Pre-LN):
```python
x = self.ln1(x + self.attn(x))
x = self.ln2(x + self.ffn(x))
```

**Benefits**:
- ✅ More stable gradients
- ✅ Often eliminates warmup scheduling
- ✅ Faster convergence
- ✅ No model size change

**File**: `state_core/operators/state_update_operator.py`  
**Test**: Compare training stability (grad norms)

**Source**: Xiong et al. (2020), standard in modern Transformers

---

### 2.2.3 Nucleus (Top-p) Sampling
**Priority**: 🟡 HIGH  
**Time**: 1-2 hours  
**Complexity**: Very Low

**What**: Replace greedy decoding with stochastic sampling

**Current**:
```python
next_token = logits.argmax()  # Always picks highest prob
```

**New**:
```python
def nucleus_sampling(logits, p=0.9, temperature=1.0):
    # Temperature scaling
    probs = F.softmax(logits / temperature, dim=-1)
    
    # Sort and compute cumulative probability
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    
    # Find nucleus (top-p)
    mask = cumsum <= p
    mask[0] = True  # Always keep top token
    
    # Zero out tail
    filtered_probs = sorted_probs * mask.float()
    filtered_probs /= filtered_probs.sum()
    
    # Sample
    next_token_idx = torch.multinomial(filtered_probs, 1)
    next_token = sorted_idx[next_token_idx]
    
    return next_token
```

**Benefits**:
- ✅ Avoids repetitive loops ("also also also...")
- ✅ More diverse, human-like text
- ✅ Standard in GPT generation

**File**: `test_long_context.py`, generation utilities  
**Test**: Generate 100 tokens, check for loops

**Source**: Holtzman et al., ICLR 2020 ("The Curious Case of Neural Text Degeneration")

---

### 2.2.4 Factorized Embeddings (ALBERT-style)
**Priority**: 🟡 HIGH  
**Time**: 3-4 hours  
**Complexity**: Low-Medium

**What**: Reduce embedding parameters via matrix factorization

**Current**:
```python
self.embeddings = nn.Embedding(vocab_size, hidden_dim)
# 50,257 × 768 = 38.6M parameters (43% of model!)
```

**New**:
```python
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, factorized_dim=128):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, factorized_dim)
        self.projection = nn.Linear(factorized_dim, hidden_dim)
    
    def forward(self, token_ids):
        # 50,257 × 128 = 6.4M
        factorized = self.word_embeddings(token_ids)
        # 128 × 768 = 0.1M
        projected = self.projection(factorized)
        return projected
```

**Benefits**:
- ✅ 6× parameter reduction (38.6M → 6.5M)
- ✅ Model: 89M → 57M parameters (36% smaller)
- ✅ Faster training (less memory)
- ✅ No quality loss (proven by ALBERT)

**File**: `state_core/adapters/mu_adapter.py`  
**Test**: Train 5 epochs, verify PPL maintained

**Source**: ALBERT (Lan et al., 2019)

---

### 2.2.5 Edge Provenance Tracking
**Priority**: 🟡 HIGH  
**Time**: 4-6 hours  
**Complexity**: Medium

**What**: Track which MU blocks contribute to each edge

**Implementation**:
```python
def _streaming_topk_with_provenance(self, semantic_state, seq_len):
    edges = []
    provenance = []
    
    # Extract block states
    I_state = _select_block(semantic_state, 'I')    # [T, 4]
    R2_state = _select_block(semantic_state, 'R2')  # [T, 4]
    K_state = _select_block(semantic_state, 'K')    # [T, 4]
    
    for i in range(seq_len):
        # Per-block similarities
       block_sims = {
            'I': F.cosine_similarity(I_state[i:i+1], I_state),
            'R2': F.cosine_similarity(R2_state[i:i+1], R2_state),
            'K': F.cosine_similarity(K_state[i:i+1], K_state)
        }
        
        # Combined (weighted or sum)
        combined = 0.4*block_sims['I'] + 0.3*block_sims['R2'] + 0.3*block_sims['K']
        
        # Top-K
        top_k_idx = combined.topk(self.semantic_k).indices
        
        for j in top_k_idx:
            edges.append((i, j.item()))
            provenance.append({
                'edge': (i, j.item()),
                'I_contrib': block_sims['I'][j].item(),
                'R2_contrib': block_sims['R2'][j].item(),
                'K_contrib': block_sims['K'][j].item(),
                'combined': combined[j].item()
            })
    
    return edges, provenance
```

**Analysis Script**:
```python
# tools/analyze_provenance.py
def analyze_edge_provenance(model, dataset):
    all_provenance = []
    
    for batch in dataset:
        _, state = model(batch, return_state=True)
        prov = state.routing_state.get('provenance', [])
        all_provenance.extend(prov)
    
    # Statistics
    block_usage = {
        'I': np.mean([p['I_contrib'] for p in all_provenance]),
        'R2': np.mean([p['R2_contrib'] for p in all_provenance]),
        'K': np.mean([p['K_contrib'] for p in all_provenance])
    }
    
    print(f"Block Utilization:")
    for block, usage in block_usage.items():
        print(f"  {block}: {usage:.3f}")
    
    return block_usage
```

**Benefits**:
- ✅ Know which blocks are actually used
- ✅ Can prune unused blocks
- ✅ Interpretability foundation
- ✅ Informs future architecture decisions

**File**: `state_core/graph/graph_builder.py`, `tools/analyze_provenance.py`  
**Test**: Run on trained model, visualize block contributions

---

## Phase 2.2 Validation

**After implementing all items**:

1. **Training Run** (10 epochs):
   ```bash
   python test_sosm.py --epochs 10
   ```

2. **Expected Results**:
   - PPL: ~3.3-3.5 (slight improvement from RoPE + Pre-LN)
   - Training speed: Similar or faster (Pre-LN helps)
   - Model size: 89M → 57M (factorized embeddings)
   - Generation: No repetition (nucleus sampling)

3. **Analysis**:
   - Edge provenance: Which blocks matter?
   - RoPE: Does it extrapolate to longer sequences?
   - Nucleus: Is generation more diverse?

**Success Criteria**:
- ✅ PPL ≤ 3.67 (no regression)
- ✅ Generation quality improved
- ✅ Training stability maintained

---

# Phase 2.3: SOSM Core Validation 🔬

**Timeline**: Week 2 (5-7 days)  
**Goal**: Test if local context helps, add interpretability  
**Risk Level**: 🟡 LOW-MEDIUM  
**Critical Decision Point**: Does context-aware refinement work?

---

## Items

### 2.3.1 3-Token Window MU Refinement
**Priority**: 🔴 CRITICAL (Decision Point!)  
**Time**: 8-10 hours  
**Complexity**: Medium-High

**What**: Add local context awareness to MU via small transformer

**Architecture**:
```python
class ContextualMURefinement(nn.Module):
    def __init__(self, mu_dim=64, window_size=3):
        super().__init__()
        # Tiny 1-layer transformer
        self.context_layer = nn.TransformerEncoderLayer(
            d_model=mu_dim,
            nhead=4,
            dim_feedforward=mu_dim * 2,
            batch_first=True
        )
        self.window_size = window_size
    
    def forward(self, mu_state):
        # mu_state: [B, T, 64]
        B, T, D = mu_state.shape
        refined = mu_state.clone()
        
        for i in range(T):
            # Extract window
            start = max(0, i - self.window_size // 2)
            end = min(T, i + self.window_size // 2 + 1)
            window = mu_state[:, start:end]  # [B, ≤3, 64]
            
            # Apply context
            context_out = self.context_layer(window)  # [B, ≤3, 64]
            
            # Update center token
            center_idx = i - start
            refined[:, i] = context_out[:, center_idx]
        
        return refined
```

**Integration**:
```python
# In pipeline.py
mu_state_base = self.mu_adapter(token_ids)  # Position-invariant
mu_state_refined = self.context_refiner(mu_state_base)  # Context-aware

# Use refined for graph building
graph = self.graph_builder.build_graph(..., mu_state_refined, ...)
```

**Expected Outcome**:
- **If homonym separation > 0.05**: Context helps! Proceed to more sophisticated approaches
- **If homonym separation < 0.01**: Context insufficient, MU is fundamentally position-invariant

**Benefits**:
- ✅ Simpler than Two-Tier Graph (rejected by peer review)
- ✅ Tests core hypothesis: does local context help?
- ✅ Minimal parameters (~200K for 1-layer, 4-head)

**File**: `state_core/adapters/contextual_refiner.py` (new)  
**Test**: Train 5 epochs, run homonym test

**Decision Point**: This determines if we pursue Phase 3 context-aware research!

---

### 2.3.2 Typed Edge Embeddings
**Priority**: 🟡 HIGH  
**Time**: 3-4 hours  
**Complexity**: Low-Medium

**What**: Different learned representations for edge types

**Implementation**:
```python
class TypedEdgeAttention(nn.Module):
    def __init__(self, hidden_dim, num_edge_types=3):
        super().__init__()
        # Edge type: 0=sequential, 1=semantic, 2=shortcut
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
    
    def forward(self, Q, K, V, edge_index, edge_types):
        # Standard attention
        scores = (Q @ K.T) / math.sqrt(K.size(-1))  # [T, T]
        
        # Add type-specific bias
        for (i, j), typ in zip(edge_index, edge_types):
            type_bias = self.edge_type_emb(typ)  # [hidden_dim]
            # Project to scalar bias
            bias = torch.dot(type_bias, Q[i])
            scores[i, j] += bias
        
        # Softmax + attention
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        
        return out
```

**Benefits**:
- ✅ Model learns edge type importance
- ✅ "Sequential → syntax, Semantic → coreference, Shortcuts → topics"
- ✅ Negligible cost (3 × hidden_dim params)
- ✅ High interpretability value

**Analysis**:
- Which edge types get highest attention?
- Do different heads specialize in different types?

**File**: `state_core/operators/state_update_operator.py`  
**Test**: Visualize attention weights per edge type

**Source**: Inspired by Graph Attention Networks, recommended by Modernization Report

---

### 2.3.3 Manual K Hyperparameter Study
**Priority**: 🟡 HIGH  
**Time**: 6-8 hours (mostly compute)  
**Complexity**: Low (just run experiments)

**What**: Test different K values to find optimal before automating

**Experiments**:
```python
k_values = [3, 5, 7, 10, 12, 15]

for k in k_values:
    config['graph']['semantic_k'] = k
    model = train(config, epochs=5)
    
    results = {
        'k': k,
        'ppl': eval_ppl(model),
        'edge_count': measure_avg_edges(model),
        'homonym_sep': homonym_test(model),
        'training_time': ...,
    }
    
    save_results(results)
```

**Metrics**:
- PPL (primary)
- Edge count (should scale with K)
- Homonym separation (does more edges help?)
- Training speed (linear in K?)

**Decision Logic**:
- If clear winner (e.g., K=10): Use that
- If no pattern: K=7 is fine (current)
- If linear improvement: Consider Adaptive K

**File**: `experiments/k_study.py` (new)  
**Test**: 6 runs × 5 epochs = 30 epoch-equivalents

**Why NOT Adaptive K yet**: Peer Review #1 recommended manual study first

---

## Phase 2.3 Validation

**Critical**: After 3-token window refinement:

**If homonym separation improves (>0.05)**:
- ✅ Context helps!
- Proceed to Phase 2.4 (SOTA integration)
- Consider Phase 3 (advanced context-aware)

**If NO improvement (<0.01)**:
- ❌ Local context insufficient
- MU is fundamentally position-invariant (by design)
- Skip context-aware research
- Focus on other improvements (provenance, typed edges, efficiency)

**This is the go/no-go decision for context-aware features!**

---

# Phase 2.4: SOTA Integration ⚡

**Timeline**: Week 3 (5-7 days)  
**Goal**: FlashAttention, WikiText-103, deterministic shortcuts  
**Risk Level**: 🟡 MEDIUM  
**Expected**: 3× training speed, 50-70% factual recall

---

## Items

### 2.4.1 FlashAttention Integration
**Priority**: 🔴 CRITICAL  
**Time**: 6-8 hours  
**Complexity**: Medium

**What**: Replace PyTorch attention with optimized FlashAttention kernel

**Installation**:
```bash
pip install flash-attn --no-build-isolation
```

**Integration**:
```python
from flash_attn import flash_attn_func

class FlashAttentionLayer(nn.Module):
    def forward(self, Q, K, V, attention_mask=None):
        # FlashAttention expects [B, T, H, D]
        # Standard shape is [B, H, T, D]
        Q = Q.transpose(1, 2)  # [B, T, H, D]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Call FlashAttention
        out = flash_attn_func(Q, K, V, causal=False)
        
        # Transpose back
        out = out.transpose(1, 2)  # [B, H, T, D]
        return out
```

**Graph Integration** (sparse attention):
```python
def flash_graph_attention(Q, K, V, edge_index):
    # Convert edge_index to block-sparse mask
    # FlashAttention supports custom masks (v2+)
    mask = create_block_sparse_mask(edge_index, block_size=64)
    
    out = flash_attn_func(Q, K, V, attn_mask=mask)
    return out
```

**Benefits**:
- ✅ 2-3× training speedup
- ✅ 40-60% memory reduction
- ✅ Enables T > 512
- ✅ Exact attention (not approximate)

**Challenges**:
- Requires CUDA 11.6+
- May need compilation
- Graph integration needs custom mask support

**File**: `state_core/operators/state_update_operator.py`  
**Test**: Benchmark PyTorch vs Flash, verify identical outputs

**Source**: Dao et al., FlashAttention (2022), FlashAttention-2 (2023)

---

### 2.4.2 WikiText-103 Training
**Priority**: 🔴 CRITICAL  
**Time**: 2-3 hours setup + overnight training  
**Complexity**: Low (just change dataset)

**What**: Train on 50× larger corpus for better factual knowledge

**Current**: WikiText-2 (2M tokens, 10MB)
**New**: WikiText-103 (103M tokens, 500MB)

**Code Change**:
```python
# In test_sosm.py or config
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')  # Changed from wikitext-2-raw-v1
```

**Expected Results**:
- Factual recall: 10% → 50-70%
- PPL: May improve slightly (more data)
- Training time: 10× longer (100 epochs → overnight)

**Benefits**:
- ✅ Fixes factual recall limitation
- ✅ No architecture change
- ✅ Proven dataset (standard benchmark)

**File**: `test_sosm.py`, data loading  
**Test**: Run `test_long_context.py` after training

**Note**: If WikiText-103 still insufficient, consider OpenWebText (8M docs)

---

### 2.4.3 Fibonacci/Power-of-2 Shortcuts
**Priority**: 🟡 MEDIUM  
**Time**: 2-3 hours  
**Complexity**: Low

**What**: Replace random shortcuts with structured long-range connections

**Current** (random):
```python
shortcuts = []
for i in range(T):
    if random.random() < shortcut_prob:
        j = random.randint(0, T-1)
        shortcuts.append((i, j))
```

**New** (Fibonacci-spaced):
```python
def fibonacci_shortcuts(T):
    fib = [1, 2]
    while fib[-1] < T:
        fib.append(fib[-1] + fib[-2])
    
    shortcuts = []
    for i in range(T):
        for f in fib:
            j = i + f
            if j < T:
                shortcuts.append((i, j))
                if bidirectional:
                    shortcuts.append((j, i))
    
    return shortcuts
```

**Alternative** (Powers-of-2):
```python
def power_of_2_shortcuts(T):
    shortcuts = []
    for i in range(T):
        k = 1
        while i + 2**k < T:
            shortcuts.append((i, i + 2**k))
            if bidirectional:
                shortcuts.append((i + 2**k, i))
            k += 1
    
    return shortcuts
```

**Benefits**:
- ✅ Deterministic (reproducible)
- ✅ O(log T) complexity (vs O(T²) betweenness)
- ✅ Structured long-range connections
- ✅ Explainable (follows logarithmic pattern)

**Why NOT Betweenness**: O(V³) too expensive (rejected by Peer Review #1)

**File**: `state_core/graph/graph_builder.py`  
**Test**: Compare random vs. Fibonacci, measure path lengths

---

## Phase 2.4 Validation

**Training Run**:
```bash
python test_sosm.py --dataset wikitext-103 --epochs 30 --flash-attention
```

**Expected Results**:
- PPL: ~3.0-3.3 (WikiText-103 may improve)
- Training time: ~3-4 hours (FlashAttention speedup)
- Factual recall: 50-70% (vs 10% on WikiText-2)
- Memory: 50% less (FlashAttention)

**Success Criteria**:
- ✅ FlashAttention 2× faster than PyTorch
- ✅ Factual recall >40%
- ✅ PPL maintained or improved

---

# Phase 2.5: Comprehensive Analysis 📊

**Timeline**: Week 4-5 (10-14 days)  
**Goal**: Understand what works, what doesn't  
**Risk Level**: 🟢 VERY LOW (just analysis)

---

## Items

### 2.5.1 Ablation Studies
**Priority**: 🔴 CRITICAL  
**Time**: 10-12 hours (mostly compute)  
**Complexity**: Medium

**MU Block Ablations** (16 tests):
```python
blocks = ['I', 'D', 'R1', 'R2', 'K', 'M', 'T', 'P', 'S', 'C', 'N', 'X', 'E', 'F', 'A', 'Z']

for block_to_remove in blocks:
    # Mask out block in similarity computation
    config['graph']['semantic_blocks'] = [b for b in ['I','R2','K'] if b != block_to_remove]
    
    model = train(config, epochs=5)
    
    results = {
        'removed_block': block_to_remove,
        'ppl': eval_ppl(model),
        'delta_ppl': ppl - baseline_ppl
    }
    
    save_results(results)
```

**Edge Type Ablations** (4 tests):
```python
configs = [
    {'sequential': True, 'semantic': False, 'shortcuts': False},  # Seq only
    {'sequential': False, 'semantic': True, 'shortcuts': False},  # Sem only
    {'sequential': True, 'semantic': True, 'shortcuts': False},   # No shortcuts
    {'sequential': True, 'semantic': True, 'shortcuts': True},    # All (baseline)
]

for config in configs:
    model = train(config, epochs=5)
    ...
```

**Hyperparameter Sweep**:
```python
thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
for threshold in thresholds:
    # Test impact of semantic threshold
    ...
```

**File**: `experiments/ablations.py` (new)  
**Output**: Comprehensive ablation report with charts

---

### 2.5.2 Gradient Flow Analysis
**Priority**: 🟡 MEDIUM  
**Time**: 4-6 hours  
**Complexity**: Medium

**What**: Document what is learned vs. frozen

**Analysis**:
```python
def analyze_gradient_flow(model):
    # Check which parameters receive gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"{name}: NO GRADIENT")
    
    # Specific to graph
    print("\nGraph Structure:")
    print("- Top-K selection: NON-DIFFERENTIABLE (hard selection)")
    print("- Threshold filtering: NON-DIFFERENTIABLE (hard mask)")
    print("- Attention weights: DIFFERENTIABLE (learned)")
    print("- MU parameters: DIFFERENTIABLE (learned via downstream)")
```

**Documentation**:
Add to `architecture_theory.md`:
```markdown
## Gradient Flow in SOSM

### What is Learned
- MU parameters (via downstream gradients)
- Attention weights
- Typed edge embeddings
- 3-token window transformer (if enabled)

### What is Frozen
- Graph structure (top-K is hard selection)
- Edge existence (binary, not differentiable)

### Why This is OK
Graph provides structural prior (routing blueprint).
Attention learns soft weights within structure.
This is intentional: separate structure from weighting.
```

**File**: `docs/gradient_flow.md` (new)

---

### 2.5.3 Runtime Profiling
**Priority**: 🟡 MEDIUM  
**Time**: 3-4 hours  
**Complexity**: Low-Medium

**What**: Measure where time is spent

**Profiling**:
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for batch in dataloader:
        loss = model(batch)
        loss.backward()

# Print profile
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Identify bottlenecks
# Is it: Graph building? Attention? MU? TEMPORAL?
```

**Decisions Based on Results**:
- If graph building slow → Implement caching
- If attention slow → Already using FlashAttention
- If MU slow → Optimize block selection
- If nothing slow → Don't optimize!

**File**: `tools/profiler.py` (new)

---

### 2.5.4 Probing Tasks (Beyond PPL)
**Priority**: 🟢 LOW (Optional)  
**Time**: 6-8 hours  
**Complexity**: Medium

**What**: Evaluate linguistic understanding

**Tasks**:
1. **POS Tagging**: Does model know parts of speech?
2. **NER**: Can it identify named entities?
3. **Dependency Parsing Probes**: Does it understand syntax?

**Implementation**:
```python
# Freeze SOSM, train linear probe on top
sosm.eval()
for param in sosm.parameters():
    param.requires_grad = False

probe = nn.Linear(sosm.hidden_dim, num_pos_tags)

for batch in pos_dataset:
    hidden = sosm(batch)[-1]  # Last layer
    logits = probe(hidden)
    loss = cross_entropy(logits, labels)
    loss.backward()
```

**File**: `experiments/probing.py` (new)

---

## Phase 2.5 Deliverables

**Comprehensive Report**:
1. Ablation study results (which blocks/edges matter?)
2. Gradient flow documentation
3. Runtime profile (bottleneck identification)
4. Probing task results (linguistic knowledge)

**Decision Points**:
- Which MU blocks can be pruned?
- Which edge types are essential?
- Where to optimize next?

---

# Phase 3: Advanced Features (Optional) 🚀

**Timeline**: Month 3-6 (if needed)  
**Goal**: Research-level contributions  
**Risk Level**: 🔴 HIGH

**Only pursue if**:
- Phase 2 results warrant further research
- Have time/resources for experiments
- Want to publish novel techniques

---

## Items (Cherry-Picked from Advanced Report #3)

### 3.1 α-Entmax (Learnable Sparsity)
**If**: Want differentiable sparse attention

**Complexity**: High (custom operator)  
**Value**: Novel, but unclear benefit for SOSM

---

### 3.2 Centrality Encoding
**If**: Want graph-aware positional biases

**Complexity**: Medium  
**Value**: Interesting, aligns with graph philosophy

---

### 3.3 AtMan (Real-Time Interpretability)
**If**: Need production explanations

**Complexity**: Low-Medium  
**Value**: High for deployment

---

### 3.4 Parameter Sharing (ALBERT)
**If**: Want even smaller model

**Complexity**: Medium  
**Value**: Proven technique

---

**Phase 3 is DEFERRED until Phase 2 complete and validated!**

---

# Future Research (Year 2+) 📚

**From Advanced Report #3 (12-month scope)**:

These are interesting but beyond current project:
- SOFT/Perturbed Top-K (Optimal Transport)
- Custom Triton Kernels
- Full Graphormer Integration
- Mixture of Graph Experts (SAGMM)
- Mixture-of-Depths
- GAF + LibraGrad Interpretability
- RAG Retrieval Integration

**These require dedicated research team and are deferred to future work.**

---

# Implementation Strategy 🎯

## Week-by-Week Plan

**Week 1**: Phase 2.2 (Modern Foundations)
- Days 1-2: RoPE + Pre-LayerNorm
- Days 3-4: Factorized Embeddings + Nucleus Sampling
- Days 5-7: Edge Provenance + Validation

**Week 2**: Phase 2.3 (SOSM Core)
- Days 1-3: 3-Token Window Refinement
- Day 4: Typed Edge Embeddings
- Days 5-7: Manual K Study

**Week 3**: Phase 2.4 (SOTA Integration)
- Days 1-2: FlashAttention
- Days 3-4: WikiText-103 Setup
- Days 5-7: Training + Fibonacci Shortcuts

**Week 4-5**: Phase 2.5 (Analysis)
- Week 4: Ablation studies
- Week 5: Profiling, probing, documentation

---

## Success Metrics

### Phase 2 Complete (5 weeks)

**Performance**:
- ✅ PPL: ≤3.5 (improved from 3.67)
- ✅ Model size: ~60M params (vs 89M)
- ✅ Training speed: 2-3× faster
- ✅ Factual recall: 50-70% (vs 10%)

**Research**:
- ✅ Know which MU blocks matter (ablations)
- ✅ Context-aware decision made (3-token test)
- ✅ Typed edges working (interpretability)
- ✅ Comprehensive documentation

**Publishable**:
- Edge provenance analysis (novel)
- Typed edge embeddings (simple but effective)
- 3-token refinement (if works)
- Comprehensive ablations

---

## Risk Mitigation

**If 3-Token Window Fails** (<0.01 separation):
- Accept MU is position-invariant
- Skip context-aware research
- Focus on efficiency + interpretability

**If FlashAttention Won't Install**:
- Use standard PyTorch attention
- Accept slower training
- Still get WikiText-103 benefits

**If WikiText-103 PPL Worse**:
- More data can hurt initially
- Train longer (50 epochs)
- Or revert to WikiText-2

---

## Summary

**This roadmap synthesizes three research reports into a practical, incremental plan:**

1. **Week 1**: Quick SOTA wins (RoPE, Pre-LN, nucleus, factorized)
2. **Week 2**: Core SOSM features (context test, typed edges)
3. **Week 3**: Scale (FlashAttention, WikiText-103)
4. **Week 4-5**: Analysis (ablations, profiling)
5. **Later**: Advanced research (α-entmax, centrality, etc.)

**Philosophy**: Start simple, validate incrementally, defer complexity until proven necessary.

**Total commitment**: 5 weeks for solid foundation, optional 3-6 months for advanced research.

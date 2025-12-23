# Self-Organizing State Model (SOSM)

A novel neural architecture that combines **semantic representation**, **temporal learning**, and **hierarchical credit assignment** into a unified self-organizing system.

---

## 🧠 Core Innovation

SOSM integrates three independent research systems into a progressive, stage-based architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SOSM Architecture Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Token IDs                                                      │
│       │                                                          │
│       ▼                                                          │
│   ┌────────────┐                                                 │
│   │    MU      │  8×8 Semantic Matrix Representation             │
│   │ (Stage 0+) │  → Structured meaning in 16 semantic blocks     │
│   └─────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│   ┌────────────┐                                                 │
│   │  TEMPORAL  │  Self-Learning Time Embeddings                  │
│   │ (Stage 1+) │  → Learns temporal patterns via gradients       │
│   └─────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│   ┌────────────┐                                                 │
│   │   Graph    │  Dynamic Attention Routing                      │
│   │ (Stage 3)  │  → Sequential, semantic, random shortcut edges  │
│   └─────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│   ┌────────────┐                                                 │
│   │Transformer │  Attention layers with graph-based masking      │
│   │  Layers    │                                                 │
│   └─────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│      Logits                                                      │
│         │                                                        │
│         ▼                                                        │
│   ┌────────────┐                                                 │
│   │    K-1     │  Hierarchical Credit Assignment                 │
│   │ (Stage 2+) │  → Sparse, interpretable gradient updates       │
│   └────────────┘                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Modules

### 1. MU (Meaning Unit)
**8×8 Semantic Matrix Representation**

| Class | Description |
|-------|-------------|
| `MU_Config` | Configuration for MU architecture |
| `MU_Transformer` | 24-layer transformer with block-wise semantic attention |
| `MU_BlockAttention` | Structure-aware attention over 16 semantic blocks |
| `MU_DynamicSensitivity` | Learned (not hardcoded) block sensitivity |

**Key Innovation:** Tokens are represented as 8×8 matrices with 16 semantic blocks:
- Identity (I): Core token meaning
- Domain (D): Domain/topic information  
- Relation (R): Relational semantics
- ...and 13 more specialized blocks

### 2. TEMPORAL
**Self-Learning Time Embeddings**

| Class | Description |
|-------|-------------|
| `Temporal_Transformer` | Production transformer with time embeddings |
| `Temporal_TimeEmbeddings` | Time embeddings that learn via gradients |
| `Temporal_Tokenizer` | Combines content + time embeddings |
| `Temporal_Config` | Production configuration |

**Key Innovation:** Time embeddings learn what "temporal experience" means through backpropagation - no manual time features needed.

### 3. K-1 (Self-Learning Hierarchical)
**Gradient-Based Credit Assignment**

| Class | Description |
|-------|-------------|
| `K1_Tree` | Hierarchical tree of transformer nodes |
| `K1_Node` | Individual node with specialization tracking |
| `K1_DataLoader` | Multi-domain data loading |

**Key Innovation:** 
- Traces errors through tree hierarchy
- Updates only responsible nodes (sparse learning)
- 100% to culprit, 15% to parent, 5% to root
- Interpretable error attribution

### 4. state_core (Integration Layer)
**Unified Execution Pipeline**

| Component | Description |
|-----------|-------------|
| `StateCorePipeline` | Main forward/backward pipeline |
| `State` | Central state object flowing through system |
| `StageController` | Stage-based component activation |
| `MUAdapter` | Wraps MU semantic representation |
| `TemporalAdapter` | Wraps TEMPORAL time embeddings |
| `K1Adapter` | Wraps K-1 gradient attribution |
| `GraphBuilder` | Builds per-sequence attention graphs |
| `GraphMaskConverter` | Converts graph to attention mask |

---

## 🎚️ Stages

SOSM supports progressive complexity through stages:

| Stage | Components | Use Case |
|-------|------------|----------|
| **0** | MU only | Baseline semantic representation |
| **1** | MU + TEMPORAL | Add temporal learning |
| **2** | MU + TEMPORAL + K-1 | Add hierarchical credit assignment |
| **3** | Full system + Graph | Add dynamic attention routing |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd Self-Organizing-State-Model

# Install dependencies
pip install torch datasets tokenizers tqdm pyyaml
```

### Run Tests

```bash
# Test baseline transformer
python test_base.py

# Test SOSM (full system)
python test_sosm.py --stage 3 --epochs 3 --batch-size 64

# Test specific stages
python test_sosm.py --stage 0  # MU only
python test_sosm.py --stage 1  # + TEMPORAL
python test_sosm.py --stage 2  # + K-1
```

### Usage in Code

```python
from state_core import StateCorePipeline
import torch

# Create pipeline (Stage 3 = full system)
config = {
    'stage': 3,
    'components': {
        'mu': {'vocab_size': 50000, 'embed_dim': 64},
        'temporal': {'time_dim': 32},
        'k1': {},
        'graph': {'sequential_edges': True, 'semantic_edges': True}
    },
    'model': {'hidden_dim': 256, 'n_layers': 6, 'n_heads': 4}
}

pipeline = StateCorePipeline(config)

# Forward pass
tokens = torch.randint(0, 50000, (batch_size, seq_len))
logits, state = pipeline(tokens)

# Backward with K-1 (Stage 2+)
loss = compute_loss(logits, labels)
loss.backward()
pipeline.backward_with_k1(loss, state, current_step)
```

---

## 📁 Project Structure

```
Self-Organizing-State-Model/
├── MU/                          # Semantic matrix module
│   ├── mu_sota.py               # MU_Transformer, MU_BlockAttention
│   └── __init__.py              # Exports: MU_Config, MU_Transformer, ...
│
├── TEMPORAL/                    # Time embedding module
│   ├── temporal_prototype/
│   │   ├── model.py             # Temporal_Transformer
│   │   ├── time_embeddings.py   # Temporal_TimeEmbeddings
│   │   └── config.py            # Temporal_Config
│   └── __init__.py              # Module exports
│
├── self-learning-k-1/           # Hierarchical learning module
│   ├── k1_system/
│   │   └── core/
│   │       ├── tree.py          # K1_Tree
│   │       └── tree_node.py     # K1_Node
│   └── __init__.py              # Module exports
│
├── state_core/                  # Integration layer
│   ├── pipeline.py              # StateCorePipeline
│   ├── state.py                 # State dataclass
│   ├── stages.py                # StageController
│   ├── adapters/                # Module adapters
│   ├── graph/                   # Graph routing
│   └── config/                  # YAML configuration
│
├── test_base.py                 # Baseline transformer test
├── test_sosm.py                 # SOSM test (all stages)
├── sosm_data.py                 # Multi-domain data loader
└── README.md                    # This file
```

---

## 🔬 Research Goals

1. **Reduce Catastrophic Forgetting**: K-1's sparse updates preserve prior knowledge
2. **Interpretable Learning**: Error attribution shows which nodes learn what
3. **Multi-Domain Specialization**: Nodes develop domain-specific expertise
4. **Temporal Pattern Discovery**: Time embeddings emerge from gradients
5. **Structured Semantics**: 8×8 matrices capture richer meaning than vectors

---

## 📊 Datasets

The test scripts automatically download:
- **WikiText-2**: General text
- **The Stack (Python)**: Code
- **ArXiv Summarization**: Scientific papers

---

## ⚙️ Configuration

Edit `state_core/config/config.yaml`:

```yaml
stage: 3  # 0-3

components:
  mu:
    enabled: true
    vocab_size: 50000
    embed_dim: 64
    
  temporal:
    enabled: true
    time_dim: 32
    
  k1:
    enabled: true
    
  graph:
    enabled: true
    sequential_edges: true
    semantic_edges: false
    random_shortcuts: 0.0
```

---

## 📈 Expected Outputs

When running `python test_sosm.py --stage 3`:

```
=== Testing Stage 3 (Full System) ===
StateCorePipeline initialized:
  Stage 3: MU + TEMPORAL + K-1 + Graph
  Model dim: 96
  Vocab size: 10000

Epoch 1/3
  Batch 0: loss=9.21, semantic=64d, temporal=32d, edges=46
  ...
  Train Loss: 7.82
  Test Loss: 7.95
  Perplexity: 284.12
  Avg Nodes Updated (K-1): 15.3
```

---

## 📝 License

MIT License

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Improved graph construction strategies
- Better K-1 attribution algorithms
- Cross-domain transfer experiments
- Scaling to larger models

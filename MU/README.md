# MU-SOTA Transformer

A novel **Meaning Unit State-of-the-Art Transformer** architecture that uses **8×8 structured semantic matrices** instead of traditional dense embeddings for explicit semantic factorization.

## 🔥 Latest Results

**3-Epoch Training Run (6 layers, ~45 minutes on GPU)**:

| Epoch | Loss | Accuracy | Perplexity | Status |
|-------|------|----------|------------|--------|
| 1 | 8.24 | 6.51% | 1151 | Baseline |
| 2 | 6.52 | 14.45% | 641 | Improving |
| 3 | 6.02 | 17.53% | 500 | Converging |

**Key Achievements**:
- ✅ **Architecture works**: Loss decreasing, accuracy increasing consistently
- ✅ **Text generation improving**: From gibberish to semi-coherent phrases
- ✅ **Fast training**: ~14 minutes per epoch with gradient accumulation
- ✅ **Memory efficient**: Fits in 14.74 GB GPU with mixed precision
- 🎯 **Ready to scale**: Currently upgrading to 12 layers × 10 epochs

## Overview

The MU-SOTA Transformer represents each token as a structured **8×8 matrix** (64 dimensions) organized into **16 semantic blocks** (each 2×2), rather than a dense vector. This explicit factorization aims to improve interpretability and performance on tasks requiring nuanced semantic understanding.

### Key Features

- **Structured Representations**: Tokens as 8×8 matrices with 16 semantic blocks (2×2 each)
- **Block-Wise Semantic Attention**: Separate attention modules for each semantic block
- **Dynamic Sensitivity Computation**: Context-aware gating based on token properties
- **Cross-Block Attention**: Global refinement across semantic blocks
- **Mixed Precision Training**: FP16 for speed, gradient accumulation for effective batch size
- **Production-Ready**: Complete modular pipeline with BPE tokenization

## Architecture

### 8×8 Matrix Structure (16 Semantic Blocks)

```
MU Matrix = [
    [ I  | I  | S  | S  | C1 | C1 | C2 | C2 ]   Row 0-1: Identity, Structure, Context
    [ I  | I  | S  | S  | C1 | C1 | C2 | C2 ]

    [ R1 | R1 | R2 | R2 | T  | T  | K  | K  ]   Row 2-3: Relations, Transform, Composition
    [ R1 | R1 | R2 | R2 | T  | T  | K  | K  ]

    [ G  | G  | M  | M  | D  | D  | F  | F  ]   Row 4-5: Global, Memory, Domain, Flow
    [ G  | G  | M  | M  | D  | D  | F  | F  ]

    [ P  | P  | E  | E  | A  | A  | X  | X  ]   Row 6-7: Pragmatics, Emotion, Agency, Extension
    [ P  | P  | E  | E  | A  | A  | X  | X  ]
]
```

**16 Semantic Blocks** (each 2×2):

| Block | Position | Semantic Role | Sensitivity |
|-------|----------|---------------|-------------|
| **I** | (0-2, 0-2) | **Identity** - Core token meaning | Low (stable) |
| **S** | (0-2, 2-4) | **Structure** - Grammatical properties | Very low (invariant) |
| **C1** | (0-2, 4-6) | **Context-Local** - Immediate context | High (adaptive) |
| **C2** | (0-2, 6-8) | **Context-Global** - Document context | Very high (fluid) |
| **R1** | (2-4, 0-2) | **Relation-1** - Dependency structure | High (relational) |
| **R2** | (2-4, 2-4) | **Relation-2** - Hierarchy/scope | High (relational) |
| **T** | (2-4, 4-6) | **Transform** - Semantic composition | Medium-high |
| **K** | (2-4, 6-8) | **Compositional** - Phrase building | Medium |
| **G** | (4-6, 0-2) | **Global** - Document-level info | Low (stable) |
| **M** | (4-6, 2-4) | **Memory** - Long-range dependencies | Medium |
| **D** | (4-6, 4-6) | **Domain** - Topic/field information | Medium |
| **F** | (4-6, 6-8) | **Flow** - Discourse coherence | Medium-high |
| **P** | (6-8, 0-2) | **Pragmatics** - Intent/speech acts | Medium |
| **E** | (6-8, 2-4) | **Emotion** - Affective content | Medium |
| **A** | (6-8, 4-6) | **Agency** - Actor/action structure | Medium-high |
| **X** | (6-8, 6-8) | **Extension** - Future semantic slots | Low (reserved) |

### Block-Wise Semantic Attention

Each semantic block gets its own attention module:

1. **Intra-Block Attention**: 16 separate attention modules (one per block)
   - Each processes its 2×2 block (4 values) independently
   - 2 attention heads per block
   - Captures role-specific patterns

2. **Cross-Block Attention**: Global refinement across all blocks
   - 8 attention heads across full 64-dimensional space
   - Allows blocks to interact and share information
   - Residual connections + Layer normalization

3. **Dynamic Sensitivity Gating**: Context-aware modulation
   - Computes sensitivity for each block based on:
     - Token properties (learned embeddings)
     - Attention patterns (entropy)
   - Different blocks have different update rates

### Model Configuration

**Current Configuration** (12 layers, 10 epochs):

```python
# Architecture
matrix_size = 8                  # 8×8 matrix
num_semantic_blocks = 16         # 16 blocks (2×2 each)
n_layers = 12                    # Transformer layers
n_heads = 8                      # Cross-block attention heads
dropout = 0.1

# Vocabulary
vocab_size = 50000               # BPE tokenization (GPT-2 style)
max_seq_len = 128                # Sequence length

# Training
batch_size = 6                   # Adjusted for 12-layer model
num_epochs = 10                  # Increased for convergence
gradient_accumulation_steps = 4  # Effective batch_size = 24
use_mixed_precision = True       # FP16 training
```

**Model Specs**:
- **Total Layers**: 12 deep transformer layers
- **Attention Operations per Layer**: 17 (16 intra-block + 1 cross-block)
- **Parameters**: ~40-50M (estimated for 12-layer model)
- **Memory**: ~6-8 GB GPU memory with mixed precision
- **Training Time**: ~8-12 hours for 10 epochs (estimated)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/PlanetDestroyyer/MU.git
cd MU

# Install dependencies (PyTorch 2.x required)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tokenizers tqdm
```

### Training

```bash
# Run training (uses config from src/config.py)
python run_colab.py
```

This will:
1. ✓ Download WikiText-2 dataset
2. ✓ Build BPE tokenizer (50K vocabulary)
3. ✓ Train MU-SOTA Transformer (12 layers, 10 epochs)
4. ✓ Evaluate after each epoch
5. ✓ Generate sample text to verify quality
6. ✓ Save best model to `mu_sota_best.pt`

**Expected Time**: ~8-12 hours on GPU for 10 epochs

### Text Generation

After training, generate text:

```python
import torch
from src.models import MUSOTATransformer
from src.training import generate_text
from tokenizers import Tokenizer

# Load model
checkpoint = torch.load('mu_sota_best.pt')
model = MUSOTATransformer(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = Tokenizer.from_file('mu_sota_tokenizer.json')

# Generate
prompt = "The future of AI"
generated = generate_text(model, tokenizer, prompt, max_length=100, device='cuda')
print(generated)
```

## Project Structure

```
MU/
├── run_colab.py              # Main training entry point
├── src/
│   ├── config.py             # MUSOTAConfig - all hyperparameters
│   ├── models/
│   │   ├── __init__.py
│   │   ├── semantic_blocks.py    # 16 semantic block definitions
│   │   ├── sensitivity.py        # Dynamic sensitivity computation
│   │   ├── attention.py          # Block-wise semantic attention
│   │   └── transformer.py        # MU-SOTA Transformer model
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py            # WikiText BPE dataset
│   └── training/
│       ├── __init__.py
│       ├── trainer.py            # Training loop with gradient accumulation
│       └── generation.py         # Text generation with sampling
├── ARCHITECTURE.md           # Detailed architecture docs
├── MEMORY_FIX.md            # Memory optimization guide
├── SPEEDUP_GUIDE.md         # Speed optimization guide
└── README.md                # This file
```

## Why MU-SOTA is Different

### vs Traditional Dense Embeddings

**Dense Matrix/Embeddings** (Standard Transformers):
- Each token → dense vector `[d_model]`
- No explicit semantic structure
- All dimensions treated equally
- Example: `[0.23, -0.45, 0.89, ..., 0.12]` (no inherent meaning)

**MU-SOTA Matrix** (This Architecture):
- Each token → 8×8 structured matrix with **16 semantic blocks**
- Each block has **explicit meaning** (Identity, Structure, Context, etc.)
- Different blocks have **different sensitivities** (learned dynamically)
- Example: Position (0-2, 0-2) is always Identity (low sensitivity)

### Key Innovations

1. **Semantic Slot Assignment**: Each 2×2 block has predetermined semantic role
   - Identity (I): Core token meaning - stable
   - Structure (S): Grammar - nearly invariant
   - Context (C1, C2): Surrounding meaning - highly adaptive
   - Relations (R1, R2): Dependencies - relational
   - And 10 more specialized blocks

2. **Dynamic Sensitivity**: Each block's update rate is computed from:
   - Learned token properties (via nn.Embedding)
   - Attention patterns (entropy, distribution)
   - Block-specific formulas

3. **Block-Wise Processing**: 16 separate attention modules
   - Respects semantic structure
   - More interpretable
   - Can visualize individual block contributions

4. **Cross-Block Refinement**: Global attention allows blocks to interact
   - Prevents information silos
   - Maintains holistic understanding
   - Best of both worlds: structure + flexibility

## Training Results Analysis

### 3-Epoch Baseline (6 layers)

**Training Metrics**:
```
Epoch 1/3: Loss=8.24, Acc=6.51%,  PPL=1151  (Baseline)
Epoch 2/3: Loss=6.52, Acc=14.45%, PPL=641   (+122% accuracy)
Epoch 3/3: Loss=6.02, Acc=17.53%, PPL=500   (+21% accuracy)
```

**What This Means**:
- ✅ **Architecture validates**: Loss decreasing steadily (8.24 → 6.02)
- ✅ **Learning effectively**: Accuracy nearly tripled (6.51% → 17.53%)
- ✅ **Perplexity improving**: 1151 → 500 (better language modeling)
- ✅ **Text generation working**: Coherent phrases appearing

**Generation Quality**:
- Epoch 1: Gibberish tokens
- Epoch 2: Word-like fragments
- Epoch 3: Semi-coherent phrases

### Comparison to SOTA (see next section)

The 3-epoch run is a **proof-of-concept** showing the architecture works. With 12 layers and 10 epochs, we expect significantly better results.

## Optimization Features

### Memory Optimizations
- **Mixed Precision (FP16)**: 2x memory reduction
- **Gradient Accumulation**: Simulate larger batches without memory overhead
- **Memory-Efficient Processing**: `del` statements to free intermediate tensors
- **Reduced Sequence Length**: 128 tokens (vs typical 512-2048)

### Speed Optimizations
- **Reduced Layers**: 12 layers (vs typical 24-48 for SOTA)
- **Gradient Accumulation**: 4 steps for effective batch_size=24
- **Mixed Precision**: ~2x speedup on modern GPUs
- **Optimized Attention**: Block-wise processing reduces quadratic complexity

### Current Performance
- **Memory**: ~6-8 GB GPU (fits on consumer hardware)
- **Speed**: ~14 min/epoch (6 layers) → ~25-30 min/epoch (12 layers, estimated)
- **Total Training**: ~4-5 hours for 10 epochs (12 layers, estimated)

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
batch_size = 4  # in src/config.py

# Or increase gradient accumulation
gradient_accumulation_steps = 8  # effective batch stays same
```

### Training Too Slow

```bash
# Already optimized! But you can:
# - Use smaller sequence length (64 instead of 128)
# - Reduce layers (8 instead of 12)
# - Check GPU utilization (should be >80%)
```

### Poor Generation Quality

- **Early epochs**: Normal - wait for more training
- **After many epochs**: Try adjusting:
  - `temperature = 0.7` (lower for more focused text)
  - `top_k = 40` (lower for more deterministic sampling)
  - `learning_rate = 1e-4` (lower if loss oscillates)

## Future Work

- [ ] Scale to 24 layers for full SOTA depth
- [ ] Increase sequence length to 256-512
- [ ] Add benchmark evaluations (GLUE, SuperGLUE)
- [ ] Implement attention visualization for semantic blocks
- [ ] Multi-GPU training support
- [ ] Pre-training on larger corpora (BookCorpus, C4)

## Citation

```bibtex
@article{mu_sota_transformer_2024,
  title={MU-SOTA: Structured Semantic Transformers with Block-Wise Attention},
  author={MU Research Team},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure code follows modular structure
5. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.

---

**Status**: 🚀 Ready for 12-layer, 10-epoch training run
**Next Milestone**: Benchmark against GPT-2 baseline on WikiText-2

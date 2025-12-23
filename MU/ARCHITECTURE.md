# MU-SOTA Modular Architecture

This document describes the refactored modular structure of the MU-SOTA Transformer implementation.

## Project Structure

```
MU/
├── src/                          # Main source code (modular)
│   ├── __init__.py              # Package initialization
│   ├── config.py                # MUSOTAConfig class
│   │
│   ├── models/                  # Model components
│   │   ├── __init__.py
│   │   ├── semantic_blocks.py   # SemanticBlockLayout (16 blocks definition)
│   │   ├── sensitivity.py       # DynamicBlockSensitivity (learned)
│   │   ├── attention.py         # BlockWiseSemanticAttention
│   │   └── transformer.py       # MUSOTATransformer (main model)
│   │
│   ├── data/                    # Data handling
│   │   ├── __init__.py
│   │   └── dataset.py           # WikiTextBPEDataset (BPE tokenization)
│   │
│   └── training/                # Training utilities
│       ├── __init__.py
│       ├── trainer.py           # train_epoch, evaluate functions
│       └── generation.py        # generate_text with sampling
│
├── run_colab.py                 # 🎯 MAIN ENTRY POINT (clean wrapper)
├── mu_sota.py                   # Original standalone implementation (for reference)
├── requirements.txt             # Dependencies (no version pinning)
└── README.md                    # Project documentation

```

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python run_colab.py
```

### Modular Imports

```python
from src.config import MUSOTAConfig
from src.models import MUSOTATransformer, SemanticBlockLayout
from src.data import WikiTextBPEDataset
from src.training import train_epoch, evaluate, generate_text

# Create config
config = MUSOTAConfig()

# Load data
dataset = WikiTextBPEDataset('train', config.max_seq_len, vocab_size=config.vocab_size)

# Create model
model = MUSOTATransformer(config)
```

## Key Components

### 1. Configuration (`src/config.py`)
- `MUSOTAConfig`: All hyperparameters and settings
- 8×8 matrix structure (16 semantic blocks)
- SOTA-level architecture (24 layers, 8 heads)
- Mixed precision training settings

### 2. Models (`src/models/`)

#### `semantic_blocks.py`
- **SemanticBlockLayout**: Defines 16 semantic blocks in 8×8 matrix
- Blocks: I, S, C1, C2, R1, R2, T, K, G, M, D, F, P, E, A, X
- Each block is 2×2 (4 values)

#### `sensitivity.py`
- **DynamicBlockSensitivity**: Computes block-wise update sensitivity
- ALL LEARNED via `nn.Parameter` - NO hardcoded values
- Token-specific affinities learned during training
- Context-dependent modulation via neural network

#### `attention.py`
- **BlockWiseSemanticAttention**: Structure-aware attention layer
- Processes each semantic block separately (16 independent attentions)
- Cross-block attention for global refinement
- Sensitivity-based gating for updates

#### `transformer.py`
- **MUSOTATransformer**: Main model class (24 layers)
- Token → 8×8 matrix embedding
- Stack of BlockWiseSemanticAttention layers
- Output projection to vocabulary

### 3. Data (`src/data/`)

#### `dataset.py`
- **WikiTextBPEDataset**: BPE tokenization (50K vocab like GPT-2)
- WikiText-2 dataset loading
- Sequence creation with overlap
- Encode/decode utilities

### 4. Training (`src/training/`)

#### `trainer.py`
- **train_epoch()**: Single epoch training with mixed precision
- **evaluate()**: Validation with perplexity calculation
- Gradient clipping and learning rate scheduling
- Progress bars with tqdm

#### `generation.py`
- **generate_text()**: Temperature/top-k/top-p sampling
- Repetition penalty to prevent loops
- Supports custom generation parameters

## Key Features

### ✅ NO Hardcoded Values
- All sensitivity values computed from learned parameters
- Block sensitivity: `nn.Parameter` learned during training
- Token affinity: Per-token, per-block learned weights
- Modulation network: Neural network computes context-dependent adjustments

### ✅ Semantic Structure
- 16 semantic blocks with specific roles:
  - **I**: Identity (core token meaning)
  - **S**: Structure (grammatical properties)
  - **C1, C2**: Context (local and global)
  - **R1, R2**: Relations (syntactic and semantic)
  - **T**: Transformation (compositional changes)
  - **K**: Knowledge (world knowledge)
  - **G**: Global (document coherence)
  - **M**: Modality (certainty/tense)
  - **D**: Discourse (rhetorical structure)
  - **F**: Frame (semantic roles)
  - **P**: Position (positional encoding)
  - **E**: Entity (named entities)
  - **A**: Affect (sentiment)
  - **X**: Extension (flexible purpose)

### ✅ SOTA Architecture
- 24 transformer layers (like GPT-2 Medium)
- 8 attention heads
- Mixed precision (FP16 + FP32)
- 50K BPE vocabulary
- Proper residual connections and layer norms

### ✅ Production-Ready
- Modular code following software engineering principles
- Proper separation of concerns
- Clear imports and dependencies
- Error handling and logging
- Type hints in function signatures

## Differences from Original

### Before (mu_sota.py)
- Single 692-line file
- All components in one place
- Harder to maintain and test

### After (Modular Structure)
- Organized into logical modules
- Each component in separate file
- Easy to import and reuse
- Follows Python best practices
- Clean entry point (run_colab.py)

## Development Workflow

1. **Modify config**: Edit `src/config.py`
2. **Change model**: Edit files in `src/models/`
3. **Update training**: Edit `src/training/trainer.py` or `generation.py`
4. **Test**: Run `python run_colab.py`

## Why This Structure?

1. **Modularity**: Each component has single responsibility
2. **Reusability**: Import only what you need
3. **Testability**: Easy to write unit tests for each module
4. **Maintainability**: Changes isolated to specific files
5. **Scalability**: Easy to add new features
6. **Best Practices**: Follows standard Python project structure

## Next Steps

Run the training:
```bash
python run_colab.py
```

This will:
1. Load WikiText-2 with BPE tokenization
2. Train MU-SOTA for 10 epochs
3. Evaluate with perplexity metrics
4. Generate sample text every 2 epochs
5. Save best model to `mu_sota_best.pt`

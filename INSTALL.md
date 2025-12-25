# Installation Guide

## Quick Start

```bash
# Clone repository
git clone https://github.com/PlanetDestroyyer/Self-Organizing-State-Model.git
cd Self-Organizing-State-Model

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

### Required
- **torch** (>=2.0.0): PyTorch for neural networks
- **transformers** (>=4.30.0): HuggingFace transformers for GPT-2 tokenizer
- **datasets** (>=2.14.0): HuggingFace datasets for WikiText, Simple Wikipedia
- **numpy** (>=1.21.0): Numerical operations
- **tokenizers** (>=0.13.0): Fast tokenization
- **pyyaml** (>=6.0): Configuration files
- **tqdm** (>=4.65.0): Progress bars

### Optional (Recommended)

#### FlashAttention (2-3× speedup)
```bash
# Requires CUDA toolkit installed
pip install flash-attn --no-build-isolation
```

**Note**: If installation fails, the code automatically falls back to standard PyTorch attention. FlashAttention provides significant speedup but can be tricky to install.

**Requirements for FlashAttention**:
- CUDA 11.6+ or 12.x
- GPU with compute capability >= 7.5 (Volta, Turing, Ampere, Ada)
- Working CUDA toolkit installation

#### Weights & Biases (experiment tracking)
```bash
pip install wandb
```

## Environment Setup

### Option 1: Conda (Recommended)
```bash
conda create -n sosm python=3.10
conda activate sosm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Option 2: venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option 3: Google Colab / Kaggle
```bash
# Colab/Kaggle already has torch installed
!git clone https://github.com/PlanetDestroyyer/Self-Organizing-State-Model.git
%cd Self-Organizing-State-Model
!pip install transformers datasets tokenizers pyyaml tqdm huggingface-hub
```

## Verifying Installation

```python
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("✓ All dependencies installed!")
```

## GPU Requirements

### Minimum
- **GPU**: NVIDIA GPU with 6GB+ VRAM (GTX 1060, RTX 2060)
- **RAM**: 16GB system RAM
- **Storage**: 10GB free space

### Recommended
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3080, RTX 4080, A100)
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space (for datasets)

### For Simple Wikipedia Training
- Minimum: 8GB VRAM
- Recommended: 16GB VRAM
- CPU Only: Possible but slow (10-20× slower)

## Common Issues

### FlashAttention Installation Fails
**Solution**: Don't worry! The code has automatic fallback to standard attention.
```python
# Code automatically detects and falls back
✗ FlashAttention not available - using standard attention
```

### CUDA Out of Memory
**Solutions**:
- Reduce batch size in config
- Reduce sequence length
- Reduce model size (hidden_dim, n_layers)
- Use gradient checkpointing

### Dataset Download Slow
**Solution**: Use a VPN or mirror, or download once and cache:
```python
from datasets import load_dataset
dataset = load_dataset('wikipedia', '20220301.simple', cache_dir='./data_cache')
```

## Testing Installation

```bash
# Quick test (2 epochs)
python test_sosm.py --epochs 2

# Expected output:
# ✅ SOSM initialized: XX.XXM parameters
# ✅ Loaded WikiText dataset
# Training starts...
```

## Updating

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Support

Having issues? Check:
1. [GitHub Issues](https://github.com/PlanetDestroyyer/Self-Organizing-State-Model/issues)
2. [Documentation](docs/)
3. [Phase Implementation Guide](docs/phases.md)

# K Hyperparameter Study

Automated experiments to find optimal `semantic_k` value for SOSM.

## Quick Start

```bash
# Full study (tests K = 3, 5, 7, 10, 12, 15)
python experiments/k_study.py --epochs 5 --device cuda

# Quick test (fewer steps)
python experiments/k_study.py --epochs 2 --max-steps 500

# Custom K values
python experiments/k_study.py --k-values 5 7 9 --epochs 3
```

## What It Does

For each K value:
1. **Trains** model for 5 epochs (1000 steps)
2. **Measures**:
   - Perplexity (PPL) - lower is better
   - Edge count - how many semantic edges created
   - Training speed - seconds per step
   - Homonym separation - context understanding (>0.05 = good)

3. **Outputs**:
   - Progress logs during training
   - Final results table
   - Recommendations for best K
   - JSON file with all data

## Example Output

```
📊 Results Summary (sorted by PPL):
K     PPL      Edges    Speed      Homonym   
--------------------------------------------------
7     3.45     142      0.023      0.067
5     3.52     118      0.021      0.052
10    3.48     167      0.026      0.071
...

🏆 Recommendations:
  Best PPL: K=7 (PPL=3.45)
  Best Speed/PPL Tradeoff: K=5 (PPL=3.52, Speed=0.021s)
  ✓ Context-aware features viable! K=10 (separation=0.071 > 0.05)
```

## Options

- `--k-values`: K values to test (default: 3 5 7 10 12 15)
- `--epochs`: Epochs per K (default: 5)
- `--max-steps`: Max steps per experiment (default: 1000)
- `--device`: cuda or cpu (default: auto-detect)
- `--output`: Output JSON file (default: k_study_results.json)

## Results File

The JSON file contains:
- Timestamp
- Configuration used
- Per-K results with PPL history, edge counts, speeds
- Homonym separation scores

Use this data to:
- Update `semantic_k` in config
- Analyze PPL vs edge count tradeoffs
- Decide if context-aware features are worth it

## CRITICAL Decision Point

The homonym separation metric determines if you should pursue context-aware features:
- **>0.05**: Context helps! Proceed with 3-token window, advanced features
- **<0.01**: Context insufficient, skip complex approaches

This is Phase 2.3's key test!

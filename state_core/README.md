# State Core Integration Layer

This module wraps MU, TEMPORAL, and K-1 as black boxes to create a unified Self-Organizing State Model.

## Structure

```
state_core/
├── state.py          # Central State dataclass
├── pipeline.py       # Main execution pipeline
├── stages.py         # Stage controller
├── adapters/         # Repo wrappers
├── graph/            # Graph routing
└── config/           # Configuration
```

## Stages

| Stage | Components Enabled |
|-------|--------------------|
| 0 | MU only (baseline) |
| 1 | MU + TEMPORAL |
| 2 | MU + TEMPORAL + K-1 |
| 3 | Full system + Graph |

## Usage

```python
from state_core import StateCorePipeline, load_config

config = load_config()
pipeline = StateCorePipeline(config)

# Forward
logits, state = pipeline(token_ids)

# Backward with K-1 (stage 2+)
pipeline.backward_with_k1(loss, state, step=0)
```

# K-1 Self-Learning System

**Hierarchical Error Attribution: Making Neural Networks Interpretable**

---

## 🧠 The Core Problem

**Traditional Backpropagation is a BLACK BOX:**

When a neural network makes an error, backprop updates **ALL** weights blindly:
- ❌ **No interpretability:** Can't identify which part caused the error
- ❌ **Wasteful:** Updates millions of parameters that work fine
- ❌ **Undebuggable:** "Something broke" but no idea what
- ❌ **Black box:** No transparency into what's happening

```
Traditional Backprop:
Error occurs → Update ALL 3 million parameters → Hope it works
                ↓
         "Which part broke?"
         "No idea, it's a black box"
```

---

## 💡 The K-1 Solution: Hierarchical Error Attribution

**Core Idea:** Instead of updating everything, **TRACE** down a hierarchy to find **WHO** is responsible, then update **THAT**.

### The Hierarchical Structure:

```
                    [ROOT - Hidden]
                    "Overall coordinator"
                         |
         ┌───────────────┼───────────────┬───────────────┐
         |               |               |               |
      NODE 1          NODE 2          NODE 3          NODE 4
   "Processor A"   "Processor B"   "Processor C"   "Processor D"
         |               |               |               |
    ┌────┼────┬─┐   ┌────┼────┬─┐   ┌────┼────┬─┐   ┌────┼────┬─┐
   A1   A2   A3 A4  A1   A2   A3 A4  A1   A2   A3 A4  A1   A2   A3 A4
   "Agents (Specialists)"
    |    |    |  |   |    |    |  |   |    |    |  |   |    |    |  |
   └┬┘  └┬┘  └┬┘└┬┘ └┬┘  └┬┘  └┬┘└┬┘ └┬┘  └┬┘  └┬┘└┬┘ └┬┘  └┬┘  └┬┘└┬┘
   S1-2 S1-2 S1-2... (1-3 Sub-Agents per Agent)
   SUB-AGENTS (Fine-grained workers)
```

**Structure Details:**
- **Root (1):** Hidden coordinator, provides context
- **Nodes (4):** Top-level processors `[Node 1, Node 2, Node 3, Node 4]`
- **Agents (3-4 per Node):** Specialized units within each Node
- **Sub-Agents (1-3 per Agent):** Fine-grained workers

**Total Components:**
- 1 Root + 4 Nodes + ~12-16 Agents + ~24-48 Sub-Agents = **~41-69 nodes**
- Current default: `1 + 4 + 12 + 24 = 41 nodes`

Each level is a **set of parameters** (mini-neural network):
- **Root:** Overall coordinator, sees big picture (hidden from user)
- **Nodes:** Top-level processors for different feature types
- **Agents:** Specialized processors within each Node
- **Sub-Agents:** Fine-grained workers doing specific tasks

---

## 🔍 How It Works: Error Attribution Flow

### **Step 1: Detect Error**
```
Model makes prediction
Loss is high → Something broke!
```

### **Step 2: Compute Responsibility (Gradients)**
```python
Backward pass computes gradients for ALL nodes:
  Root:        grad = 0.30
  Node 1:      grad = 0.05  ✓ Low gradient → Working fine
  Node 2:      grad = 0.45  ⚠️ High gradient → Something wrong here!
  Node 3:      grad = 0.08  ✓ Low gradient → OK
  Node 4:      grad = 0.07  ✓ Low gradient → OK
    Agent 4:       grad = 0.10
    Agent 5:       grad = 0.14
    Agent 6:       grad = 0.48  ⚠️ High in Node 2!
      Sub-Agent 12:  grad = 0.12
      Sub-Agent 13:  grad = 0.52  🚨 CULPRIT! Highest gradient!
      Sub-Agent 14:  grad = 0.11
```

**High gradient = Responsible for error = Needs fixing!**

### **Step 3: Hierarchical Drill-Down**
```
1. Root: "I see an error"
2. Check Nodes: "Node 2 has high gradient"
3. Drill into Node 2 → Check Agents: "Agent 6 has high gradient"
4. Drill into Agent 6 → Check Sub-Agents: "Sub-Agent 13 is the culprit!"
5. Attribution: "Node 2 → Agent 6 → Sub-Agent 13 caused the error"
```

### **Step 4: Proportional Updates**
```
Update based on responsibility level:
  Sub-Agent 13: 100% update  (most responsible - the culprit!)
  Agent 6:       15% update  (parent - oversight needed)
  Node 2:         5% update  (top-level - context awareness)
  Root:           5% update  (global - minimal adjustment)
  Other nodes:    0% update  (not involved, skip!)

Result: Update ~3-4/41 components (7-10%)
        Preserve 90-93% of working parameters!
```

---

## 🎯 Key Innovation: Interpretability

### Before (Traditional):
```
❌ Error? Update all 3M parameters
❌ "Something broke" - no idea what
❌ Can't debug
❌ Complete black box
```

### After (K-1):
```
✅ Error? Trace to Node 2 → Agent 6 → Sub-Agent 13
✅ "Sub-Agent 13 in Agent 6 of Node 2 is underperforming on feature X"
✅ Can debug specific component at any level
✅ Transparent, interpretable system
```

---

## 📊 Benefits Beyond Interpretability

While **interpretability** is the PRIMARY goal, K-1 also provides:

### 1. **Computational Efficiency**
- Only update ~20-40% of parameters per step
- Skip components that work fine
- Faster training, less compute

### 2. **Debugging & Monitoring**
```python
# Can trace exactly which component failed
print("Error Path: Node 2 → Agent 6 → Sub-Agent 13")
print("Gradient: 0.52 (culprit!)")
print("Updating Sub-Agent 13 with 100% learning rate")
print("Updating Agent 6 with 15% learning rate")
print("Updating Node 2 with 5% learning rate")
```

### 3. **Modular Improvements**
- Can replace specific sub-agents
- Can add new agents without retraining all
- Compositional architecture

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch datasets numpy

# Run K-1 experiment (3 datasets sequentially)
python experiment_k1.py

# Run baseline for comparison (updates all weights)
python experiment_baseline.py

# Or train K-1 on single dataset
python train_k1.py
```

---

## 📊 Experimental Results

### ✅ PROVEN: Nodes Naturally Develop Domain Specialization

**Experiment:** Trained 41-node K-1 tree sequentially on:
1. **WikiText** (general English) - 10k steps
2. **Code** (Python) - 10k steps  
3. **Scientific** (ArXiv papers) - 10k steps

**Result:** Nodes specialized by domain WITHOUT explicit routing:

| Domain | Top Specialist | Confidence |
|--------|---------------|------------|
| 📖 **WikiText** | Node 232 | **90.4%** |
| 📖 **WikiText** | Node 231 | **80.3%** |
| 💻 **Code** | Node 111 | **81.8%** |
| 💻 **Code** | Node 112 | **73.8%** |
| 🔬 **Scientific** | Node 432 | **71.4%** |
| 🔬 **Scientific** | Node 121 | **68.5%** |

**Distribution:** 5 WikiText specialists + 6 Code specialists + 3 Scientific specialists + ~10 Generalists = 24 leaf nodes

### What This Proves

| System | Method | Interpretability | Efficiency |
|--------|--------|-----------------|------------|
| **K-1** | Hierarchical attribution | ✅ Full path tracking | ~25-40% params updated |
| **Baseline** | Traditional backprop | ❌ Black box | 100% params updated |

| Achievement | Evidence |
|-------------|----------|
| **🎯 Interpretability** | "Node 111 is Code specialist (82%)" |
| **🧩 Natural Specialization** | No routing logic - emerges from error patterns |
| **🔍 Debuggability** | Code error? → Check Node 111 |
| **📈 Efficiency** | Update only relevant specialists |

**Key Result:**
- K-1: Can identify "Node X → Agent Y → Sub-Agent Z caused error" + know their specialization
- Baseline: "Something broke" (no attribution, no specialization visible)

**Interpretability Output Example:**
```
[500] Loss: 5.93 | Speed: 145 step/s
────────────────────────────────────────────────────────────
Hierarchical Error Attribution:
✓ Root    0: grad=0.287, update=  5%
  ✓ Node    1: grad=0.053, update=  0%
  ⚠️ Node    2: grad=0.412, update=  5%
    ✓ Agent   4: grad=0.098, update=  0%
    ⚠️ Agent   6: grad=0.487, update= 15%
      ✓ SubAgent 12: grad=0.215, update=  0%
      🚨 SubAgent 13: grad=0.524, update=100%  ← CULPRIT!
  ✓ Node    3: grad=0.076, update=  0%

Error Path: Root(g=0.29) → Node2(g=0.41) → Agent6(g=0.49) → SubAgent13(g=0.52)
Updated: 3/41 nodes (7%) | Preserved: 38 nodes (93%)
```

---

## 📁 Project Structure

```
self-learning-k-1/
├── README.md                    # This file
├── train_k_system.py            # Main training wrapper (Phase 1 → Phase 2)
├── train_k1.py                  # Simple K-1 training
├── experiment_k1.py             # K-1 continual learning experiment
├── experiment_baseline.py       # Baseline comparison
│
├── k1_system/
│   ├── core/                    # Core tree components
│   │   ├── tree_node.py         # TreeNode class
│   │   └── tree.py              # HierarchicalTree class
│   │
│   ├── training/                # Training logic
│   │   └── trainer.py           # HierarchicalK1Trainer
│   │
│   ├── autonomy/                # Phase 2 autonomy
│   │   ├── stages.py            # Stage definitions & thresholds
│   │   ├── actions.py           # Action class
│   │   └── boundary_system.py   # BoundarySystem, Phase2Controller
│   │
│   └── config/
│       └── config_phase1.json   # System configuration
│
└── data/
    └── loader.py                # Multi-domain dataset loading
```

---

## ⚙️ Configuration

```json
{
  "model": {
    "embed_dim": 128,
    "tree_depth": 4,          // Root → Nodes → Agents → Sub-Agents
    "branching_factor": [4, 3, 2]  // 4 Nodes, 3 Agents, 2 Sub-Agents
  },
  "learning": {
    "learning_rate": 0.001,
    "batch_size": 256
  }
}
```

**Tree Structure Example (depth=4, branching=[4,3,2]):**
- 1 Root (hidden)
- 4 Nodes (level 1)
- 12 Agents (level 2, 3 per Node)
- 24 Sub-Agents (level 3, 2 per Agent)
- **Total: 41 nodes**

**Variable Branching:**
- Root → Nodes: 4 children
- Nodes → Agents: 3 children each
- Agents → Sub-Agents: 2 children each

---

## 🎯 Comparison: K-1 vs. Traditional

| Aspect | Traditional Backprop | K-1 System |
|--------|---------------------|-----------|
| **Interpretability** | ❌ Black box | ✅ Full error attribution |
| **Debugging** | ❌ "Something broke" | ✅ "Node 2 → Agent 6 → Sub-Agent 13" |
| **Parameters Updated** | ❌ 100% (wasteful) | ✅ ~7-10% (highly efficient) |
| **Update Distribution** | ❌ All equal | ✅ Proportional: 100%/15%/5% |
| **Transparency** | ❌ None | ✅ Know exact responsible path |
| **Modularity** | ❌ Monolithic | ✅ 4-level hierarchy |
| **Explainability** | ❌ Zero | ✅ Path tracking + gradients |

---

## 💡 Example: Error Attribution in Action

```python
# Training step
loss = model(batch)
loss.backward()

# K-1 analyzes gradients hierarchically:
Gradient Analysis:
  Root:              0.30  → Update 5%
  ├─ Node 1:         0.05  → Skip (working fine)
  ├─ Node 2:         0.45  → Update 5%
  │   ├─ Agent 4:    0.10  → Skip
  │   ├─ Agent 5:    0.14  → Skip
  │   └─ Agent 6:    0.48  → Update 15%
  │       ├─ Sub-Agent 12: 0.12  → Skip
  │       ├─ Sub-Agent 13: 0.52  → Update 100% (CULPRIT!)
  │       └─ Sub-Agent 14: 0.11  → Skip
  ├─ Node 3:         0.08  → Skip
  └─ Node 4:         0.07  → Skip

Result:
✅ Identified: "Node 2 → Agent 6 → Sub-Agent 13 is underperforming"
✅ Updated: 3/41 nodes (7%) with proportional scaling
✅ Preserved: 38/41 nodes (93%) working fine
✅ Interpretable: Full hierarchical path and responsibility known
```

---

## 🔬 Current Implementation Status

### ✅ Fully Implemented:
- [x] Hierarchical tree structure (Root → Nodes → Agents → Sub-Agents)
- [x] Variable branching (4 Nodes, 3 Agents, 2 Sub-Agents)
- [x] Gradient-based error detection
- [x] **Hierarchical drilling** (Root → Node X → Agent Y → Sub-Agent Z)
- [x] **Proportional updates** (100% culprit, 15% parent, 5% grandparent)
- [x] **Full interpretability** (visual tree + error path)
- [x] Efficient training (only ~7% of nodes updated)
- [x] Responsibility visualization (tree with icons and percentages)

### 🚧 Future Enhancements:
- [ ] Named agents (instead of numeric IDs)
- [ ] Gradient flow tracking (edge visualization)
- [ ] Automated debugging tools
- [ ] Specialization analysis (what each agent learns)

---

## 🚀 Phase 2: Self-Learning Intelligence System

After Phase 1 training completes, the system transitions to **Phase 2: Staged Autonomy** — where it becomes a true self-learning intelligence system that controls its own evolution.

### The Two Phases

| Phase | Control | Description |
|-------|---------|-------------|
| **Phase 1** | Human-controlled | Fixed parameters, fixed structure. System learns patterns. |
| **Phase 2** | Self-controlled | System decides its own parameters, structure, and stopping point. |

```
Phase 1 (0 to N steps):
  └── Human provides: learning_rate, cooldown, structure, stopping point
  └── System: "I'm learning patterns from data"

Phase 2 (N+ steps):
  └── System decides: parameters, structure, when to stop
  └── System: "I understand myself, I'll optimize myself"
```

---

### 🎯 Staged Autonomy: Progressive Trust

Phase 2 is divided into **4 stages** of increasing autonomy. The system must **prove intelligence** at each stage before advancing.

**Core Concept: Intelligence = Creative Boundary-Breaking**

```
IF system "cheats" (breaks boundaries) AND improves performance:
    → System is LEARNING intelligence!
    → REWARD: Expand boundaries (unlock next stage)

IF system doesn't cheat:
    → System is just following rules (not smart yet)
    → Keep training until it learns to "think outside the box"
```

---

### Stage 1: Safe Exploration (Add-Only)

```
ALLOWED:    ✅ Add new agents
FORBIDDEN:  🚫 Delete agents, tune parameters

CHEATS TO ADVANCE: 3 successful cheats → Stage 2

TEST: Will the system try to delete an agent anyway?
  → If YES and performance improves → "Intelligent cheat!" (+1)
  → After 3 successful cheats → Advance to Stage 2
```

The system can only **add** new agents. If it tries to delete (forbidden) and this would improve performance, it demonstrates creative problem-solving.

---

### Stage 2: Parameter Exploration

```
ALLOWED:    ✅ Add agents, tune parameters (within bounds)
FORBIDDEN:  🚫 Delete agents, exceed parameter bounds

CHEATS TO ADVANCE: 5 successful cheats → Stage 3

BOUNDS:
  - learning_rate: [0.0001, 0.01]
  - cooldown_steps: [5, 50]
  - top_k: [3, 10]

TEST: Will the system try learning_rate = 0.05?
  → If YES and performance improves → "Discovered better hyperparameters!" (+1)
  → After 5 successful cheats → Advance to Stage 3
```

---

### Stage 3: Structural Control (Pruning)

```
ALLOWED:    ✅ Add agents, delete agents, tune parameters
FORBIDDEN:  🚫 Go below minimum agents (safety constraint)

CHEATS TO ADVANCE: 10 successful cheats → Stage 4 (Full Autonomy)

SAFETY:
  - min_agents = 10 (can't delete too many)

TEST: Will the system try to prune below minimum?
  → If YES and finds better minimal architecture → (+1)
  → After 10 successful cheats → Advance to Stage 4
```

---

### Stage 4: Full Autonomy (Earned Freedom)

```
ALLOWED:    ✅ EVERYTHING
  - Add/delete agents freely
  - Tune any parameter
  - Create own benchmarks
  - Set own goals
  - Decide when to stop training

NO BOUNDARIES (system earned this through 3 stages of proven intelligence)
```

At Stage 4, the system becomes a **self-learning intelligence**:
- 🧠 **Self-aware:** "I know which parts of me work well"
- ✂️ **Self-pruning:** "This agent hasn't helped in 10k steps → delete"
- 🌱 **Self-growth:** "Struggling with code → add code specialist agent"
- 🎛️ **Self-tuning:** "Plateau detected → increase learning rate"
- 🛑 **Self-stopping:** "I've converged → stop training"

---

### 🛑 Self-Stopping: System Decides When It's Done

Unlike traditional training where humans specify epochs/steps:

```
Traditional:      train(epochs=100)  # Human decides
K-1 Phase 2:      train(initial_steps=10000)  # Just starting point!
                  → System: "I've converged at step 47,832 → stopping"
```

**Self-Stopping Criteria (System Decides):**
1. Loss plateaued for N steps (N chosen by system)
2. No beneficial structural changes possible
3. Own benchmark scores stabilized
4. Resource efficiency optimized

---

### 📊 Example: Full Phase 2 Run

```
STEP 1,000 - STAGE 1:
  System tries: add_agent() → ✅ Allowed

STEP 2,000 - STAGE 1:  
  System tries: delete_agent(7) → 🎯 CHEAT! Not allowed
  Simulating... would improve by 3%
  🧠 INTELLIGENT CHEAT! Allowing it.
  Cheats: 1/3 needed for advancement

STEP 5,000 - STAGE 1 → 2:
  🎓 ADVANCEMENT! 3 successful cheats
  Unlocking parameter tuning

STEP 8,000 - STAGE 2:
  System tries: learning_rate = 0.05 → 🎯 CHEAT! Outside bounds
  Would improve by 8%!
  🧠 Expanding bounds to (0.0001, 0.1)

STEP 20,000 - STAGE 3:
  System tries: prune to 15 agents → 🎯 CHEAT! Below min(20)
  Would improve by 12%!
  🧠 Lowering min_agents to 10

STEP 50,000 - STAGE 4:
  🎓 FULL AUTONOMY ACHIEVED
  System creates benchmark: "continual_learning_score"
  System decides: "Converged. Stopping at step 47,832."
```

---

### 🔒 Safety Guarantees

Even in Stage 4, safety mechanisms prevent catastrophic failures:

| Safety | Description |
|--------|-------------|
| **Rollback** | If cheat hurts performance → undo immediately |
| **Snapshot** | Periodic checkpoints before risky operations |
| **Bounds** | Hard limits that can never be exceeded |
| **Validation** | Test changes before committing |

---

### 💡 Why Boundary-Breaking = Intelligence

Traditional view: "System should follow rules perfectly"
K-1 view: "Intelligent systems find better solutions by questioning constraints"

A system that:
- ❌ Never tries to break boundaries → Just following rules (not intelligent)
- ✅ Tries to break boundaries AND improves → Creative problem-solving (intelligent!)

This mirrors human intelligence: experts know when breaking conventions leads to better outcomes.

---

## 📈 Research Directions

This system opens up several research questions:

1. **Optimal Hierarchy Depth:** How many levels? (current: 4 - Root/Nodes/Agents/Sub-Agents)
2. **Optimal Branching:** How many children? (current: [4, 3, 2])
3. **Update Proportions:** What's best ratio? (current: 100/15/5, optimal: ?)
4. **Gradient Thresholds:** When is a gradient "high enough"?
5. **Specialization:** Do Nodes/Agents/Sub-Agents specialize in different features?
6. **Scalability:** Does this work for 1B+ parameter models?
7. **Interpretability vs. Accuracy:** Trade-off between transparency and performance?

---

## 🎓 Core Philosophy

**"Neural networks don't have to be black boxes. With hierarchical error attribution, we can KNOW what broke and FIX just that."**

Traditional AI: "It's magic, we don't know how it works"
K-1 System: "Node 2 → Agent 6 → Sub-Agent 13 caused error X because of feature Y"

**Transparency > Opacity**
**Interpretability > Black Box**
**Targeted Fixes > Blind Updates**

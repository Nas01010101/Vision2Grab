# Phase 1: Simulation & Learning Foundations

**Goal**: Master imitation learning algorithms in MuJoCo simulation.

---

## Structure

```
phase1_sim/
├── algorithms/       # BC, DAgger, IDM implementations
│   ├── train_bc.py   # Behavior Cloning training
│   └── eval.py       # Policy evaluation
├── envs/             # MuJoCo environments
│   └── mujoco_env.py # Env wrappers and utilities
└── training/         # Training configs and scripts
```

---

## Milestones

- [ ] **M1.1** Train BC on Walker2d, achieve >30% expert performance
- [ ] **M1.2** Implement DAgger with expert querying
- [ ] **M1.3** Implement IDM for unlabeled data labeling
- [ ] **M1.4** Create custom pick-and-place MuJoCo scene
- [ ] **M1.5** Video logging and metrics dashboard

---

## Key Concepts

| Algorithm | Description |
|-----------|-------------|
| **BC** | Supervised learning on (s, a) pairs |
| **DAgger** | Iteratively query expert to fix distribution shift |
| **IDM** | Predict actions from (s, s') to label unlabeled data |

---

## Getting Started

```bash
# Train BC
python -m src.phase1_sim.algorithms.train_bc --dataset data/demos.npz

# Evaluate
python -m src.phase1_sim.algorithms.eval --checkpoint runs/bc/best_policy.pt
```

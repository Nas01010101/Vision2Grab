# Vision2Grab — Imitation Learning for Robotics

**Dirobots** | Universite de Montreal Robotics Club  
**Project W26** | Winter 2026

---

## Description

Imitation learning project for robotic manipulation. The goal is to train policies in simulation, then deploy them on a real robot arm.

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Simulation and learning (BC, DAgger) | In Progress |
| **Phase 2** | Real robot integration | Upcoming |
| **Phase 3** | Sim-to-real transfer | Upcoming |

---

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train BC (Phase 1)
python -m src.phase1_sim.algorithms.train_bc --dataset data/demos.npz
```

---

## Project Structure

```
src/
├── phase1_sim/           # Simulation-based learning
│   ├── algorithms/       # BC, DAgger, IDM
│   ├── envs/             # MuJoCo environments
│   └── training/         # Training scripts
│
├── phase2_robot/         # Real robot code
│   ├── drivers/          # Robot interface
│   ├── teleop/           # Demo collection
│   └── deploy/           # Policy deployment
│
├── phase3_sim2real/      # Sim-to-real transfer
│   ├── domain_rand/      # Domain randomization
│   ├── system_id/        # System identification
│   └── distillation/     # Model compression
│
└── shared/               # Shared code
    ├── policy.py         # Neural networks
    ├── dataset.py        # Data loading
    └── utils/            # Utilities
```

---

## Team

**Project Leads:** Anas & Quan  
**Simulation:** Pierre  
**Coding & Model Fine-tuning:** Josh

---

## License

Academic project — Dirobots, Universite de Montreal, 2026.

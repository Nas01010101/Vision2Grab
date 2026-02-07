# Project Roadmap

Brouillon

## Phase 1: Simulation & Learning (Current)

### Algorithms
- [x] Implement BC training loop
- [x] Policy network (MLP)
- [x] Dataset loading (.npz, .pkl)
- [x] Evaluation metrics
- [ ] DAgger implementation
- [ ] IDM for unlabeled data

### Environments
- [x] MuJoCo env wrappers
- [ ] Custom pick-and-place scene
- [ ] Reaching task scene

### Tooling
- [x] Logging utilities
- [x] Config management
- [x] Reproducibility (seeding)
- [ ] Weights & Biases integration
- [ ] Video logging

---

## Phase 2: Real Robot (Upcoming)

### Hardware Interface
- [ ] Robot arm driver
- [ ] Camera integration
- [ ] Gripper control

### Data Collection
- [ ] Teleoperation system
- [ ] Demo recording format
- [ ] Calibration tools

### Deployment
- [ ] Real-time policy runner
- [ ] Safety limits
- [ ] Performance benchmarks

---

## Phase 3: Sim2Real (Future)

### Domain Randomization
- [ ] Physics randomization
- [ ] Visual randomization
- [ ] Action noise

### Transfer
- [ ] System identification
- [ ] Policy distillation
- [ ] Sim2real benchmarks

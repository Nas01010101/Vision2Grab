# Phase 3: Sim2Real Transfer

**Goal**: Train in simulation, deploy on real robot with minimal performance gap.

---

## Structure

```
phase3_sim2real/
├── domain_rand/      # Randomization during training
│   └── randomizer.py # Physics, visual randomization
├── system_id/        # Match sim to real
│   └── tuner.py      # Parameter optimization
└── distillation/     # Compress policies
    └── distill.py    # Teacher-student training
```

---

## Milestones

- [ ] **M3.1** Implement physics domain randomization (mass, friction, damping)
- [ ] **M3.2** Visual randomization (textures, lighting, backgrounds)
- [ ] **M3.3** Action noise injection for robustness
- [ ] **M3.4** System identification: tune sim to match real dynamics
- [ ] **M3.5** Policy distillation for real-time inference
- [ ] **M3.6** Evaluate sim2real gap on benchmark tasks

---

## Key Techniques

| Technique | Description |
|-----------|-------------|
| **Domain Randomization** | Vary sim parameters to cover real-world distribution |
| **System ID** | Tune sim physics to match real robot dynamics |
| **Visual Randomization** | Random textures, lighting, camera poses |
| **Distillation** | Compress large policy into smaller real-time model |

---

## Key Papers

- [Domain Randomization for Sim2Real](https://arxiv.org/abs/1703.06907) - OpenAI
- [SimOpt: System Identification](https://arxiv.org/abs/1810.05687)
- [Learning Dexterous Manipulation](https://arxiv.org/abs/1808.00177) - OpenAI Rubik's Cube

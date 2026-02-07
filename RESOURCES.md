# Ressources Theoriques / Theory Resources


## Imitation Learning

### Behavior Cloning (BC)
- [A Reduction of Imitation Learning to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) — Ross et al., 2011 (DAgger paper)
- [Learning from Demonstrations Survey](https://arxiv.org/abs/1606.03476) — Argall et al.
- [Berkeley Deep RL: Imitation Learning](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf)

### DAgger (Dataset Aggregation)
- [Original DAgger Paper](https://arxiv.org/abs/1011.0686)
- [Interactive Imitation Learning](https://www.ri.cmu.edu/pub_files/2014/3/ICRA2014-Ross.pdf)

### Inverse Dynamics Model (IDM)
- [Video PreTraining (VPT)](https://arxiv.org/abs/2206.11795) — OpenAI, 2022
- [Learning from Unlabeled Video](https://openai.com/research/vpt)

---

## Sim2Real Transfer

### Domain Randomization
- [Domain Randomization for Sim2Real](https://arxiv.org/abs/1703.06907) — Tobin et al., 2017
- [Learning Dexterous In-Hand Manipulation](https://arxiv.org/abs/1808.00177) — OpenAI Rubik's Cube

### System Identification
- [SimOpt](https://arxiv.org/abs/1810.05687) — Chebotar et al.
- [Closing the Sim-to-Real Loop](https://arxiv.org/abs/1810.05687)

---

## MuJoCo

### Documentation
- [MuJoCo Official Docs](https://mujoco.readthedocs.io/)
- [Gymnasium MuJoCo Envs](https://gymnasium.farama.org/environments/mujoco/)
- [dm_control Suite](https://github.com/google-deepmind/dm_control)

### Tutorials
- [MuJoCo Tutorial](https://pab47.github.io/mujoco.html)
- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)

---

## PyTorch

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)
- [nn.Module Guide](https://pytorch.org/docs/stable/nn.html)

---

## Robot Learning Courses

- [MILA Robot Learning 2025](https://github.com/milarobotlearningcourse/robot_learning_2025) — Glen Berseth
- [Berkeley Deep RL](http://rail.eecs.berkeley.edu/deeprlcourse/) — Sergey Levine
- [Stanford CS229M](https://cs229m.stanford.edu/)
- [CMU Robot Learning](https://www.cs.cmu.edu/~rsalleb/10701/)

---

## Libraries

| Library | Usage | Link |
|---------|-------|------|
| PyTorch | Neural networks | https://pytorch.org |
| Gymnasium | Environments | https://gymnasium.farama.org |
| MuJoCo | Physics simulation | https://mujoco.org |
| Stable-Baselines3 | RL algorithms | https://stable-baselines3.readthedocs.io |
| imitation | IL library | https://github.com/HumanCompatibleAI/imitation |
| robomimic | Robot manipulation | https://robomimic.github.io |

---

## Papers to Read (Priority)

1. **DAgger** — Core algorithm for iterative IL
2. **VPT** — IDM for labeling unlabeled data
3. **Domain Randomization** — Sim2Real basics
4. **BC-Z** — Zero-shot behavior cloning

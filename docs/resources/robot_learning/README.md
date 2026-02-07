# Robot Learning Resources

## Core Concepts

**Imitation Learning** - Learning from demonstrations instead of reward signals.

**Behavior Cloning (BC)** - Supervised learning on (state, action) pairs from expert demos.

**DAgger** - Dataset Aggregation: iteratively query expert to fix distribution shift.

**Inverse Dynamics Model (IDM)** - Predict actions from (state, next_state) pairs.

---

## Key Papers

- [A Reduction of Imitation Learning to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) - DAgger (Ross et al., 2011)
- [Video PreTraining (VPT)](https://arxiv.org/abs/2206.11795) - IDM + BC at scale (OpenAI, 2022)
- [Learning from Demonstrations](https://arxiv.org/abs/1606.03476) - Survey (Argall et al.)

---

## Courses

- [MILA Robot Learning 2025](https://github.com/milarobotlearningcourse/robot_learning_2025) - Glen Berseth
- [Berkeley Deep RL](http://rail.eecs.berkeley.edu/deeprlcourse/) - Sergey Levine
- [Stanford CS229M](https://cs229m.stanford.edu/) - Machine Learning Theory

---

## Code References

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [imitation](https://github.com/HumanCompatibleAI/imitation) - Imitation learning library
- [robomimic](https://github.com/ARISE-Initiative/robomimic) - Robot manipulation from demos

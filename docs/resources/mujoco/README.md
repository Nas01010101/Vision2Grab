# MuJoCo Resources

## What is MuJoCo?

Multi-Joint dynamics with Contact. A physics engine for robotics simulation.

---

## Installation

```bash
pip install mujoco gymnasium
```

---

## Common Environments

| Environment | Obs Dim | Act Dim | Description |
|-------------|---------|---------|-------------|
| `Hopper-v4` | 11 | 3 | Single-leg hopping |
| `Walker2d-v4` | 17 | 6 | Bipedal walking |
| `HalfCheetah-v4` | 17 | 6 | Running cheetah |
| `Ant-v4` | 27 | 8 | Quadruped locomotion |
| `Humanoid-v4` | 376 | 17 | Full humanoid |

---

## Quick Start

```python
import gymnasium as gym

env = gym.make("Walker2d-v4", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## Documentation

- [MuJoCo Docs](https://mujoco.readthedocs.io/)
- [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/)
- [dm_control](https://github.com/google-deepmind/dm_control) - DeepMind's control suite

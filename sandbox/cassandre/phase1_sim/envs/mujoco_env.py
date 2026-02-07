"""
MuJoCo Environment Utilities

Wrappers and helpers for MuJoCo-based gymnasium environments.
"""

from typing import Any, Optional, Tuple
import numpy as np


def make_env(
    env_id: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> Any:
    """
    Create a gymnasium MuJoCo environment.

    Args:
        env_id: Environment ID (e.g., "Walker2d-v4", "HalfCheetah-v4")
        seed: Random seed for reproducibility.
        render_mode: "human", "rgb_array", or None.

    Returns:
        Gymnasium environment.
    """
    import gymnasium as gym
    
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_env_info(env: Any) -> dict:
    """Get observation and action dimensions from environment."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high
    
    return {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "act_low": act_low,
        "act_high": act_high,
    }


class NormalizeObservation:
    """Wrapper to normalize observations online."""
    
    def __init__(self, env: Any, epsilon: float = 1e-8):
        self.env = env
        self.epsilon = epsilon
        self.obs_mean = np.zeros(env.observation_space.shape)
        self.obs_var = np.ones(env.observation_space.shape)
        self.count = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_stats(obs)
        return self._normalize(obs), reward, terminated, truncated, info

    def _update_stats(self, obs: np.ndarray) -> None:
        """Update running mean and variance."""
        self.count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.count
        self.obs_var += delta * (obs - self.obs_mean)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.obs_var / max(self.count, 1)) + self.epsilon
        return (obs - self.obs_mean) / std

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class ClipAction:
    """Wrapper to clip actions to valid range."""
    
    def __init__(self, env: Any):
        self.env = env
        self.low = env.action_space.low
        self.high = env.action_space.high

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, self.low, self.high)
        return self.env.step(action)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


# Available MuJoCo environments
MUJOCO_ENVS = {
    "walker": "Walker2d-v4",
    "cheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "ant": "Ant-v4",
    "humanoid": "Humanoid-v4",
    "pendulum": "InvertedPendulum-v4",
    "double_pendulum": "InvertedDoublePendulum-v4",
}

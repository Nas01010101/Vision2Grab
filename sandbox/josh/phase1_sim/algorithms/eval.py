"""
Policy Evaluation Module

Evaluate trained policies in MuJoCo environments.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import argparse

import numpy as np
import torch

from src.shared.policy import MLPPolicy


def evaluate_policy(
    policy: torch.nn.Module,
    env: Any,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False,
    deterministic: bool = True,
    normalize_obs: bool = False,
    obs_mean: Optional[np.ndarray] = None,
    obs_std: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate a trained policy in an environment.

    Args:
        policy: Trained policy network.
        env: Gymnasium environment.
        num_episodes: Number of evaluation episodes.
        max_steps: Max steps per episode.
        render: Whether to render.
        deterministic: Use deterministic actions.
        normalize_obs: Whether to normalize observations.
        obs_mean: Observation mean for normalization.
        obs_std: Observation std for normalization.

    Returns:
        Dictionary of evaluation metrics.
    """
    policy.eval()
    
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        print(obs) # See the starting observation
        episode_return = 0.0
        episode_length = 0
        
        for step in range(max_steps):
            # Normalize observation if needed
            if normalize_obs and obs_mean is not None and obs_std is not None:
                obs_normalized = (obs - obs_mean) / obs_std
            else:
                obs_normalized = obs

            # Get action from policy
            action = policy.get_action(obs_normalized.astype(np.float32))
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1

            if render:
                env.render()

            if terminated or truncated:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    metrics = {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "min_return": float(np.min(episode_returns)),
        "max_return": float(np.max(episode_returns)),
        "num_episodes": num_episodes,
    }

    return metrics


def collect_demonstrations(
    env: Any,
    policy: torch.nn.Module,
    num_episodes: int = 10,
    max_steps: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    Collect demonstrations by running a policy.

    Args:
        env: Gymnasium environment.
        policy: Policy to collect from.
        num_episodes: Number of episodes.
        max_steps: Max steps per episode.

    Returns:
        Dictionary with observations, actions, rewards.
    """
    policy.eval()
    
    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        
        for step in range(max_steps):
            action = policy.get_action(obs.astype(np.float32))
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(terminated or truncated)
            
            obs = next_obs
            if terminated or truncated:
                break

    return {
        "observations": np.array(all_obs, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "dones": np.array(all_dones, dtype=bool),
    }


def load_policy_checkpoint(
    checkpoint_path: str,
) -> MLPPolicy:
    """Load a policy from checkpoint."""
    return MLPPolicy.load(checkpoint_path)


def main() -> None:
    """Entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate BC policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint")
    parser.add_argument("--env", type=str, default="Walker2d-v4", help="Gym environment")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    import gymnasium as gym
    
    policy = load_policy_checkpoint(args.checkpoint)
    env = gym.make(args.env, render_mode="human" if args.render else None)
    
    metrics = evaluate_policy(policy, env, num_episodes=args.episodes, render=args.render)
    
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")

    env.close()


if __name__ == "__main__":
    main()

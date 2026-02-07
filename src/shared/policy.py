"""
Policy Network Definitions

This module defines the policy architectures for imitation learning.
A policy maps observations to actions: π(s) → a
"""

from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Tuple[int, ...] = (256, 256),
    activation: type = nn.ReLU,
) -> nn.Sequential:
    """Build a multi-layer perceptron."""
    layers: List[nn.Module] = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class MLPPolicy(nn.Module):
    """
    Multi-Layer Perceptron policy for continuous control.

    Args:
        obs_dim: Dimension of observation space.
        act_dim: Dimension of action space.
        hidden_dims: Tuple of hidden layer sizes.
        activation: Activation function class.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: type = nn.ReLU,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.network = build_mlp(obs_dim, act_dim, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass: observation → action."""
        return self.network(obs)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action for deployment (numpy in/out)."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float()
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            action = self.forward(obs_tensor)
            return action.squeeze(0).numpy()

    def save(self, path: str) -> None:
        """Save policy checkpoint."""
        torch.save({
            "state_dict": self.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "MLPPolicy":
        """Load policy from checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        policy = cls(
            obs_dim=checkpoint["obs_dim"],
            act_dim=checkpoint["act_dim"],
            **kwargs
        )
        policy.load_state_dict(checkpoint["state_dict"])
        return policy

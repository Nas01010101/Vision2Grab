"""Smoke tests for the phased project structure."""

import numpy as np
import torch


def test_shared_policy():
    """Test that shared policy works."""
    from src.shared.policy import MLPPolicy
    
    policy = MLPPolicy(obs_dim=10, act_dim=4)
    obs = torch.randn(32, 10)
    actions = policy(obs)
    assert actions.shape == (32, 4)


def test_shared_dataset():
    """Test shared dataset."""
    from src.shared.dataset import DemoDataset
    
    obs = np.random.randn(100, 10).astype(np.float32)
    act = np.random.randn(100, 4).astype(np.float32)
    dataset = DemoDataset(observations=obs, actions=act)
    assert len(dataset) == 100


def test_shared_utils():
    """Test shared utilities."""
    from src.shared.utils.seed import set_seed
    from src.shared.utils.config import get_default_bc_config
    from src.shared.utils.logging import setup_logger
    
    set_seed(42)
    config = get_default_bc_config()
    assert "training" in config


def test_phase1_imports():
    """Test phase 1 module imports."""
    from src.phase1_sim.algorithms import train_bc, eval
    from src.phase1_sim.envs import mujoco_env


def test_phase2_imports():
    """Test phase 2 module imports."""
    from src.phase2_robot import drivers, teleop, deploy


def test_phase3_imports():
    """Test phase 3 module imports."""
    from src.phase3_sim2real import domain_rand, system_id, distillation

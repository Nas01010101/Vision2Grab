"""
Random Seed Utilities

This module provides utilities for reproducible experiments.

Responsibilities:
    - Set seeds for all random number generators
    - Ensure reproducibility across runs
    - Handle GPU determinism settings

Usage:
    from src.utils.seed import set_seed
    set_seed(42)  # Set all seeds for reproducibility
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    This sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value.
        deterministic: If True, set PyTorch to deterministic mode.
            Note: This may impact performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_random_seed() -> int:
    """
    Generate a random seed.

    Returns:
        Random integer seed.
    """
    return random.randint(0, 2**32 - 1)


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize random seeds for DataLoader workers.

    Use this with DataLoader(worker_init_fn=worker_init_fn).

    Args:
        worker_id: DataLoader worker ID.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

"""
Dataset Handling for Imitation Learning

This module handles loading, processing, and batching expert demonstrations.
"""

from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DemoDataset(Dataset):
    """
    PyTorch Dataset for expert demonstrations.

    Supports .npz and .pkl formats with keys: observations, actions
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        observations: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> None:
        if data_path is not None:
            self.observations, self.actions = self._load_data(data_path)
        elif observations is not None and actions is not None:
            self.observations = observations.astype(np.float32)
            self.actions = actions.astype(np.float32)
        else:
            raise ValueError("Provide either data_path or (observations, actions)")

        self.normalize = normalize
        self.obs_mean = np.zeros(self.observations.shape[1], dtype=np.float32)
        self.obs_std = np.ones(self.observations.shape[1], dtype=np.float32)
        self.act_mean = np.zeros(self.actions.shape[1], dtype=np.float32)
        self.act_std = np.ones(self.actions.shape[1], dtype=np.float32)

        if self.normalize:
            self._compute_normalization_stats()

    def _load_data(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load demonstrations from file."""
        path = Path(path)
        
        if path.suffix == ".npz":
            data = np.load(path)
            obs = data["observations"].astype(np.float32)
            act = data["actions"].astype(np.float32)
        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Handle list of episodes or flat arrays
            if isinstance(data, list):
                obs = np.concatenate([ep["observation"] for ep in data]).astype(np.float32)
                act = np.concatenate([ep["action"] for ep in data]).astype(np.float32)
            else:
                obs = data["observations"].astype(np.float32)
                act = data["actions"].astype(np.float32)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        return obs, act

    def _compute_normalization_stats(self) -> None:
        """Compute mean/std for normalization."""
        self.obs_mean = self.observations.mean(axis=0)
        self.obs_std = self.observations.std(axis=0) + 1e-8
        self.act_mean = self.actions.mean(axis=0)
        self.act_std = self.actions.std(axis=0) + 1e-8

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using dataset stats."""
        return (obs - self.obs_mean) / self.obs_std

    def denormalize_act(self, act: np.ndarray) -> np.ndarray:
        """Denormalize actions to original scale."""
        return act * self.act_std + self.act_mean

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self.observations[idx].copy()
        act = self.actions[idx].copy()

        if self.normalize:
            obs = (obs - self.obs_mean) / self.obs_std
            act = (act - self.act_mean) / self.act_std

        return torch.from_numpy(obs), torch.from_numpy(act)

    @property
    def obs_dim(self) -> int:
        return self.observations.shape[1]

    @property
    def act_dim(self) -> int:
        return self.actions.shape[1]


def create_dataloader(
    dataset: DemoDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for the demonstration dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_test_split(
    dataset: DemoDataset,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DemoDataset, DemoDataset]:
    """Split dataset into train and test sets."""
    np.random.seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)
    split = int(n * train_ratio)

    train_obs = dataset.observations[indices[:split]]
    train_act = dataset.actions[indices[:split]]
    test_obs = dataset.observations[indices[split:]]
    test_act = dataset.actions[indices[split:]]

    train_ds = DemoDataset(observations=train_obs, actions=train_act, normalize=dataset.normalize)
    test_ds = DemoDataset(observations=test_obs, actions=test_act, normalize=dataset.normalize)
    
    # Share normalization stats
    test_ds.obs_mean = train_ds.obs_mean
    test_ds.obs_std = train_ds.obs_std
    test_ds.act_mean = train_ds.act_mean
    test_ds.act_std = train_ds.act_std

    return train_ds, test_ds

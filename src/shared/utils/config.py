"""
Configuration Management

This module handles experiment configuration loading and management.

Responsibilities:
    - Load YAML/JSON config files
    - Merge configs with command-line overrides
    - Validate configuration values
    - Provide default configurations
    - Save config snapshots with experiments

Usage:
    from src.utils.config import load_config, save_config
    config = load_config("configs/bc_default.yaml")
    config = update_config(config, {"lr": 0.001})
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.

    Supports YAML and JSON formats.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        if path.suffix in [".yaml", ".yml"]:
            # TODO: Add yaml import when needed
            # import yaml
            # return yaml.safe_load(f)
            raise NotImplementedError("YAML loading not yet implemented")
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a file.

    Args:
        config: Configuration dictionary.
        save_path: Path to save configuration.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def get_default_bc_config() -> Dict[str, Any]:
    """
    Get default configuration for Behavior Cloning.

    Returns:
        Default BC configuration.
    """
    return {
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        "policy": {
            "hidden_dims": [256, 256],
            "activation": "relu",
        },
        "data": {
            "normalize": True,
            "train_split": 0.8,
        },
        "logging": {
            "log_interval": 100,
            "save_interval": 1000,
            "use_wandb": False,
        },
    }


def update_config(
    base_config: Dict[str, Any],
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update configuration with overrides (nested merge).

    Args:
        base_config: Base configuration.
        updates: Updates to apply.

    Returns:
        Updated configuration.
    """
    result = base_config.copy()

    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = update_config(result[key], value)
        else:
            result[key] = value

    return result

"""
Logging Utilities

This module provides logging infrastructure for experiments.

Responsibilities:
    - Console logging with configurable levels
    - File logging for experiment tracking
    - Integration with Weights & Biases (wandb)
    - TensorBoard support
    - Metric aggregation and reporting

Usage:
    from src.utils.logging import setup_logger, log_metrics
    logger = setup_logger("bc_training")
    log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name.
        log_dir: Directory for log files. If None, console only.
        level: Logging level.
        use_wandb: Whether to initialize wandb.
        wandb_project: Wandb project name.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # TODO: Initialize wandb if requested
    if use_wandb and wandb_project:
        pass  # wandb.init(project=wandb_project)

    return logger


def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    logger: Optional[logging.Logger] = None,
    use_wandb: bool = False,
) -> None:
    """
    Log metrics to console, file, and optionally wandb.

    Args:
        metrics: Dictionary of metric names and values.
        step: Training step number.
        logger: Logger instance.
        use_wandb: Whether to log to wandb.
    """
    # Format metrics string
    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())

    if logger:
        logger.info(f"Step {step} | {metrics_str}")
    else:
        print(f"Step {step} | {metrics_str}")

    # TODO: Log to wandb
    if use_wandb:
        pass  # wandb.log(metrics, step=step)

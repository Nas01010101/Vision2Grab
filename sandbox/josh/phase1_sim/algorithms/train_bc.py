"""
Behavior Cloning Training Script

Train a policy to imitate expert demonstrations via supervised learning.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.shared.policy import MLPPolicy
from src.shared.dataset import DemoDataset, create_dataloader, train_test_split
from src.shared.utils.seed import set_seed
from src.shared.utils.logging import setup_logger, log_metrics


def train_bc(
    dataset_path: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dims: tuple = (256, 256),
    checkpoint_dir: Optional[str] = None,
    eval_interval: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a Behavior Cloning policy.

    Args:
        dataset_path: Path to the expert demonstration dataset.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        hidden_dims: Hidden layer dimensions for policy.
        checkpoint_dir: Directory to save checkpoints.
        eval_interval: Evaluate every N epochs.
        seed: Random seed.

    Returns:
        Dictionary with training results.
    """
    set_seed(seed)
    logger = setup_logger("bc_training")
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = DemoDataset(dataset_path, normalize=True)
    train_ds, test_ds = train_test_split(dataset, train_ratio=0.8, seed=seed)
    
    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = create_dataloader(test_ds, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    logger.info(f"Obs dim: {train_ds.obs_dim}, Act dim: {train_ds.act_dim}")

    # Initialize policy
    policy = MLPPolicy(
        obs_dim=train_ds.obs_dim,
        act_dim=train_ds.act_dim,
        hidden_dims=hidden_dims,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    logger.info(f"Using device: {device}")

    # Training setup
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training loop
    best_test_loss = float("inf")
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        # Train
        policy.train()
        train_losses = []
        
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred_actions = policy(obs_batch)
            loss = loss_fn(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Evaluate
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            policy.eval()
            test_losses = []
            
            with torch.no_grad():
                for obs_batch, act_batch in test_loader:
                    obs_batch = obs_batch.to(device)
                    act_batch = act_batch.to(device)
                    pred_actions = policy(obs_batch)
                    loss = loss_fn(pred_actions, act_batch)
                    test_losses.append(loss.item())

            avg_test_loss = np.mean(test_losses)
            history["test_loss"].append(avg_test_loss)

            log_metrics(
                {"train_loss": avg_train_loss, "test_loss": avg_test_loss},
                step=epoch + 1,
                logger=logger,
            )

            # Save best checkpoint
            if checkpoint_dir and avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                save_path = Path(checkpoint_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                policy.save(str(save_path / "best_policy.pt"))
                logger.info(f"Saved best policy (test_loss={avg_test_loss:.6f})")

    # Save final checkpoint
    if checkpoint_dir:
        policy.save(str(Path(checkpoint_dir) / "final_policy.pt"))

    return {
        "policy": policy,
        "history": history,
        "train_dataset": train_ds,
        "test_dataset": test_ds,
    }


def main() -> None:
    """Entry point for BC training."""
    parser = argparse.ArgumentParser(description="Train Behavior Cloning policy")
    parser.add_argument("--dataset", type=str, required=True, help="Path to demo dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default="runs/bc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = train_bc(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    
    print(f"Training complete! Final test loss: {results['history']['test_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()

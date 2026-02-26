import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from src.phase1_sim.envs.mujoco_env import make_env, MUJOCO_ENVS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper", choices=MUJOCO_ENVS.keys())
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="runs/ppo")
    args = parser.parse_args()

    env_id = MUJOCO_ENVS[args.env]
    env = make_env(env_id=env_id, seed=args.seed, render_mode=None)

    out_dir = Path(args.out) / args.env
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "ppo_expert.zip"

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(str(model_path))

    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    env.close()

    print(f"Saved expert to: {model_path}")
    print(f"Eval return mean={mean_r:.2f} std={std_r:.2f}")


if __name__ == "__main__":
    main()
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from src.phase1_sim.envs.mujoco_env import make_env, MUJOCO_ENVS
from src.shared.policy import MLPPolicy
from src.phase1_sim.algorithms.train_bc import train_bc


def eval_bc(env, policy: MLPPolicy, episodes: int = 10, seed: int = 42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    policy.eval()

    rets = []
    lens = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        steps = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act = policy(obs_t).squeeze(0).cpu().numpy()
            obs, r, term, trunc, _ = env.step(act)
            total += float(r)
            steps += 1
            done = term or trunc
        rets.append(total)
        lens.append(steps)
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(lens))


def load_npz(path: str):
    d = np.load(path)
    return d["observations"].astype(np.float32), d["actions"].astype(np.float32)


def save_npz(path: str, obs: np.ndarray, act: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, observations=obs.astype(np.float32), actions=act.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper", choices=MUJOCO_ENVS.keys())
    parser.add_argument("--expert", type=str, default="runs/ppo/hopper/ppo_expert.zip")
    parser.add_argument("--init_dataset", type=str, default="data/demos_ppo.npz")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rollout_steps", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="runs/dagger")
    args = parser.parse_args()

    env_id = MUJOCO_ENVS[args.env]
    env = make_env(env_id=env_id, seed=args.seed, render_mode=None)

    # Load expert PPO (oracle)
    expert = PPO.load(args.expert)

    # Start dataset
    obs_all, act_all = load_npz(args.init_dataset)

    out_dir = Path(args.out_dir) / args.env
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for it in range(args.iters + 1):
        # Save current aggregated dataset
        dataset_path = str(out_dir / f"dagger_dataset_iter{it}.npz")
        save_npz(dataset_path, obs_all, act_all)

        # Train BC on aggregated dataset
        ckpt_dir = str(out_dir / f"bc_iter{it}")
        bc_res = train_bc(
            dataset_path=dataset_path,
            epochs=args.epochs,
            batch_size=64,
            lr=1e-3,
            hidden_dims=(256, 256),
            checkpoint_dir=ckpt_dir,
            eval_interval=10,
            seed=args.seed,
        )
        bc_policy = bc_res["policy"]

        # Evaluate BC
        mean_r, std_r, mean_len = eval_bc(env, bc_policy, episodes=10, seed=args.seed)
        results.append((it, obs_all.shape[0], mean_r, std_r, mean_len))
        print(f"[Iter {it}] dataset={obs_all.shape[0]} return_mean={mean_r:.2f} std={std_r:.2f} len_mean={mean_len:.1f}")

        # Last iter: stop (don’t collect more)
        if it == args.iters:
            break

        # Collect rollouts with current BC policy, label with expert actions
        new_obs = []
        new_act = []

        obs, _ = env.reset(seed=args.seed + 1000 + it)
        for t in range(args.rollout_steps):
            # BC action (to drive into its state distribution)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                bc_act = bc_policy(obs_t).squeeze(0).cpu().numpy()

            # Expert label for this obs
            exp_act, _ = expert.predict(obs, deterministic=True)

            new_obs.append(obs)
            new_act.append(exp_act)

            obs, r, term, trunc, _ = env.step(bc_act)
            if term or trunc:
                obs, _ = env.reset()

        new_obs = np.asarray(new_obs, dtype=np.float32)
        new_act = np.asarray(new_act, dtype=np.float32)

        # Aggregate
        obs_all = np.concatenate([obs_all, new_obs], axis=0)
        act_all = np.concatenate([act_all, new_act], axis=0)

    env.close()

    # Save summary
    summary_path = out_dir / "dagger_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("iter,dataset_size,return_mean,return_std,episode_len_mean\n")
        for it, n, m, s, l in results:
            f.write(f"{it},{n},{m:.6f},{s:.6f},{l:.6f}\n")

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
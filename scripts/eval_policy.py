import argparse
import numpy as np
import torch

from src.phase1_sim.envs.mujoco_env import make_env, MUJOCO_ENVS
from src.shared.policy import MLPPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper", choices=MUJOCO_ENVS.keys())
    parser.add_argument("--policy", type=str, default="runs/bc/best_policy.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="Render (may be slow)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_id = MUJOCO_ENVS[args.env]
    render_mode = "human" if args.render else None
    env = make_env(env_id=env_id, seed=args.seed, render_mode=render_mode)

    # Load policy (dims are inside checkpoint)
    policy = MLPPolicy.load(args.policy, hidden_dims=(256, 256))
    policy.to(device)
    policy.eval()

    returns = []
    lengths = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total = 0.0
        steps = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act = policy(obs_t).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(act)
            total += float(reward)
            steps += 1
            done = terminated or truncated

        returns.append(total)
        lengths.append(steps)

    env.close()

    print(f"Env: {args.env} ({env_id})")
    print(f"Episodes: {args.episodes}")
    print(f"Return mean={np.mean(returns):.2f} std={np.std(returns):.2f}")
    print(f"Len mean={np.mean(lengths):.1f} std={np.std(lengths):.1f}")
    print("Returns:", [round(r, 2) for r in returns])


if __name__ == "__main__":
    main()
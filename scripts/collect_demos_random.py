import argparse
import os
import numpy as np

from src.phase1_sim.envs.mujoco_env import make_env, MUJOCO_ENVS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper", choices=MUJOCO_ENVS.keys())
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--out", type=str, default="data/demos_random.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    env_id = MUJOCO_ENVS[args.env]
    env = make_env(env_id=env_id, seed=args.seed, render_mode=None)

    obs, _ = env.reset(seed=args.seed)
    observations = []
    actions = []

    for _ in range(args.steps):
        action = env.action_space.sample()
        observations.append(obs)
        actions.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    np.savez(args.out, observations=observations, actions=actions)
    print("Saved:", args.out)
    print("observations:", observations.shape, "actions:", actions.shape)


if __name__ == "__main__":
    main()
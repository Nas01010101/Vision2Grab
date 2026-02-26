import os
import numpy as np

OBS_DIM = 24
ACT_DIM = 4
N = 20000

os.makedirs("data", exist_ok=True)

observations = np.random.randn(N, OBS_DIM).astype(np.float32)
actions = np.random.randn(N, ACT_DIM).astype(np.float32)

np.savez("data/demos.npz", observations=observations, actions=actions)

print("Saved -> data/demos.npz")
print("observations:", observations.shape)
print("actions:", actions.shape)
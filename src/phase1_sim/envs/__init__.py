"""MuJoCo simulation environments."""

from gymnasium.envs.registration import register

register(
    id='RX150-v0',
    entry_point='src.phase1_sim.envs.rx150_env:RX150Env',
    max_episode_steps=500,
)

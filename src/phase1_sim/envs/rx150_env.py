import os
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class RX150Env(MujocoEnv, utils.EzPickle):
    """
    Custom MuJoCo environment for the Interbotix Reactor X150 arm.
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        
        # Determine path to the xml file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "assets", "rx150", "rx150.xml")
        
        # 5 joints + 2 gripper joints = 7 action dims
        # Observations: joint positions (7), joint velocities (7), target pos (3) = 17
        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config={},
            **kwargs,
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        # Simple distance to target reward
        eef_pos = self.data.site_xpos[self.model.site("eef_site").id] if hasattr(self.model, 'site') else self.data.site_xpos[self.model.site_name2id("eef_site")]
        target_pos = self.data.geom_xpos[self.model.geom("target_geom").id] if hasattr(self.model, 'geom') else self.data.geom_xpos[self.model.geom_name2id("target_geom")]
        dist = np.linalg.norm(eef_pos - target_pos)
        
        reward = -dist
        info = {
            "distance_to_target": dist
        }
        
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, False, False, info

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()  # Positions (includes target free joint)
        qvel = self.data.qvel.flat.copy()  # Velocities
        
        # qpos structure: [waist, shoulder, elbow, wrist_angle, wrist_rotate, left_finger, right_finger, target_x, target_y, target_z, target_qw, target_qx, target_qy, target_qz]
        robot_qpos = qpos[:7]
        robot_qvel = qvel[:7]
        target_pos = qpos[7:10]
        
        return np.concatenate([robot_qpos, robot_qvel, target_pos])

    def reset_model(self):
        # Reset to initial position plus some noise
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.05, high=0.05, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        
        # Randomize target position
        qpos[7:9] = self.np_random.uniform(low=-0.2, high=0.2, size=2) + np.array([0.3, 0.0])
        qpos[9] = 0.1 # z height
        
        self.set_state(qpos, qvel)
        return self._get_obs()

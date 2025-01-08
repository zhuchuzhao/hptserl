import time
import gymnasium
import numpy as np
from gymnasium.spaces import Box, flatten_space, flatten
from mujoco_sim.devices.input_utils import input2action  # Relative import for input2action
from mujoco_sim.devices.keyboard import Keyboard  # Relative import from devices.keyboard
from mujoco_sim.devices.mujoco_keyboard import MujocoKeyboard  # Relative import from devices.mujoco_keyboard
from mujoco_sim.devices.spacemouse import SpaceMouse  # Relative import from devices.spacemouse

    
class ObsWrapper(gymnasium.ObservationWrapper):
    """
    This observation wrapper treats the observation space as a dictionary
    of a flattened state space and optionally the images, if available.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gymnasium.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        # Flatten the state observation space
        state_space = flatten_space(self.proprio_space)

        if "images" in self.env.observation_space.spaces:
            self.observation_space = gymnasium.spaces.Dict(
                {
                    "state": state_space,
                    **(self.env.observation_space["images"]),
                }
            )
            self.include_images = True
            print("Images included in observation space.")
        else:
            # If no image observations
            self.observation_space = gymnasium.spaces.Dict(
                {
                    "state": state_space,
                }
            )
            self.include_images = False

    def observation(self, obs):
        # Flatten the state observation
        if self.include_images:
            obs = {
                "state": flatten(
                    self.proprio_space,
                    {key: obs["state"][key] for key in self.proprio_keys},
                ),
                **(obs["images"]),
            }
        else:
            obs = {
                "state": flatten(
                    self.proprio_space,
                    {key: obs["state"][key] for key in self.proprio_keys},
                ),
            }

        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info



class GripperCloseEnv(gymnasium.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        if action.shape[0] == 6:
            # Policy action without gripper value
            new_action[:6] = action.copy()
        elif action.shape[0] == 7:
            # Action includes gripper value (e.g., from SpacemouseIntervention)
            new_action = action.copy()
        else:
            raise ValueError(f"Unexpected action shape: {action.shape}")
        new_action[6] = 1  # Ensure the gripper is closed
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info

class XYZGripperCloseEnv(gymnasium.ActionWrapper):
    """
    Wrapper to reduce action space to x, y, z deltas.
    """
    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:3], ub.high[:3])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        if action.shape[0] == 3:
            # Policy action with x, y, z translations
            new_action[:3] = action.copy()
        elif action.shape[0] == 7:
            # Full action provided (from SpacemouseIntervention)
            new_action = action.copy()
        else:
            raise ValueError(f"Unexpected action shape: {action.shape}")
        # Zero out rotations
        new_action[3:6] = 0
        new_action[6] = 1  # Ensure the gripper is closed
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
class XYZQzGripperCloseEnv(gymnasium.ActionWrapper):
    """
    Wrapper to reduce action space to x, y, z deltas.
    """
    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        # Set the new action space to x, y, z translations and z-axis rotation
        low = np.concatenate([ub.low[:3], ub.low[5:6]], axis=0)
        high = np.concatenate([ub.high[:3], ub.high[5:6]], axis=0)
        self.action_space = Box(
            low=low,
            high=high,
            dtype=np.float32
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        if action.shape[0] == 4:
            # Map the reduced action back to the full action space
            new_action[:3] = action[:3]      # x, y, z translations
            new_action[5] = action[3]        # z-axis rotation
        elif action.shape[0] == 7:
            # Full action provided (from SpacemouseIntervention)
            new_action = action.copy()
        else:
            raise ValueError(f"Unexpected action shape: {action.shape}")
        new_action[3:5] = 0    # Set the gripper to closed
        new_action[6] = 1      # Set the gripper to closed
        return new_action
    
    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info

class SpacemouseIntervention(gymnasium.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        try:
            # Attempt to initialize the SpaceMouse
            self.expert = SpaceMouse()
            self.expert.start_control()
            print("SpaceMouse connected successfully.")
        except OSError:
            # If SpaceMouse is not found, fall back to Keyboard
            print("SpaceMouse not found, falling back to Keyboard.")
            # self.expert = Keyboard()
            self.expert = MujocoKeyboard()
            viewer = env.unwrapped._viewer.viewer
            viewer.set_external_key_callback(self.expert.external_key_callback)

        self.expert.start_control()
        self.last_intervene = 0


    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        action = input2action(self.expert)

        if action is None:
            # Handle reset signal
            return None, False

        if np.linalg.norm(action) > 0.001:
            self.last_intervene = time.time()
            
        if time.time() - self.last_intervene < 0.5:
            return action, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)
        if new_action is None:
            # Reset the environment
            obs, info = self.env.reset()
            return obs, 0.0, False, False, info 
        
        else:
            obs, rew, done, truncated, info = self.env.step(new_action)
            if replaced:
                info["intervene_action"] = new_action
            return obs, rew, done, truncated, info

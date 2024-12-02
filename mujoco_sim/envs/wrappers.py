import time
import gym
import numpy as np
from gym.spaces import Box
from mujoco_sim.devices.input_utils import input2action  # Relative import for input2action
from mujoco_sim.devices.keyboard import Keyboard  # Relative import from devices.keyboard
from mujoco_sim.devices.spacemouse import SpaceMouse  # Relative import from devices.spacemouse


class CustomObsWrapper(gym.ObservationWrapper):
    """
    Removal of unwanted coordinates before flattening.
    """

    def __init__(self, env):
        super().__init__(env)

        # Specify the keys you want to keep in the observation
        self.keys_to_keep = {
            # "ur5e/tcp_pose",
            "ur5e/tcp_vel",
            # "ur5e/gripper_pos",
            # "ur5e/joint_pos",
            # "ur5e/joint_vel",
            "ur5e/wrist_force",
            "ur5e/wrist_torque",
            # "connector_pose"
        }

        # Modify the observation space to include only the desired keys
        original_state_space = self.observation_space["state"]
        # Efficiently filter the observation space
        modified_state_space = gym.spaces.Dict({
            key: space for key, space in original_state_space.spaces.items()
            if key in self.keys_to_keep
        })

        self.observation_space = gym.spaces.Dict({"state": modified_state_space})

    def observation(self, observation):
        # Keep only the desired keys in the observation
        observation["state"] = {
            key: observation["state"][key] for key in self.keys_to_keep
        }
        # print(observation["state"])
        return observation

import gym
from gym.spaces import flatten_space, flatten

class ObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treats the observation space as a dictionary
    of a flattened state space
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.env.observation_space["state"]),
            }
        )

    def observation(self, obs):

        obs = {
            "state": flatten(self.env.observation_space["state"], obs["state"]),
        }
        return obs
    

class GripperCloseEnv(gym.ActionWrapper):
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

class XYZGripperCloseEnv(gym.ActionWrapper):
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
    
class XYZQzGripperCloseEnv(gym.ActionWrapper):
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

class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        try:
            # Attempt to initialize the SpaceMouse
            self.expert = SpaceMouse(pos_sensitivity=0.1, rot_sensitivity=0.2)
            self.expert.start_control()
            print("SpaceMouse connected successfully.")
        except OSError:
            # If SpaceMouse is not found, fall back to Keyboard
            print("SpaceMouse not found, falling back to Keyboard.")
            self.expert = Keyboard(pos_sensitivity=0.03, rot_sensitivity=5)

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

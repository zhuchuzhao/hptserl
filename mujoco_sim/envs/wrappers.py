import time
import gymnasium
import numpy as np
from collections import deque
from typing import Optional
import jax
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
        self.action_space = Box(ub.low[:6], ub.high[:6],
            dtype=np.float32)

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
        self.action_space = Box(ub.low[:3], ub.high[:3],
            dtype=np.float32)

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

        self.expert.start_control()
        self.last_intervene = 0
        self.prev_grasp = None  # To track the previous grasp state


    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        action_ = input2action(self.expert)

        if action_ is None:
            # Handle reset signal
            return None, False

        if np.linalg.norm(action_[:-1]) > 0.0001 :
            self.last_intervene = time.time()
            self.prev_grasp = action_[-1]  # Update the previous grasp state
            
        if time.time() - self.last_intervene < 0.5:
            return action_, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)
        # print(f"New action: {new_action}")
        if new_action is None:
            # Reset the environment
            obs, info = self.env.reset()
            return obs, 0.0, False, False, info 
        
        else:
            obs, rew, done, truncated, info = self.env.step(new_action)
            if replaced:
                info["intervene_action"] = new_action
                # print("Intervened with action")
            return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and attach the keyboard callback (if needed)."""
        obs, info = super().reset(**kwargs)

        # Now that the environment is reset, the viewer should exist.
        if isinstance(self.expert, MujocoKeyboard):
            render_mode = getattr(self.env.unwrapped, "render_mode", None)
            if render_mode == "human":
                viewer = getattr(self.env.unwrapped._viewer, "viewer", None)
                # Make sure the viewer actually has 'set_external_key_callback'
                if viewer is not None and hasattr(viewer, "set_external_key_callback"):
                    viewer.set_external_key_callback(self.expert.external_key_callback)
                viewer = self.env.unwrapped._viewer.viewer
                viewer.set_external_key_callback(self.expert.external_key_callback)

        return obs, info
    
def stack_obs(obs):
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return jax.tree_map(
        lambda x: np.stack(x), dict_list, is_leaf=lambda x: isinstance(x, list)
    )


def space_stack(space: gymnasium.Space, repeat: int):
    if isinstance(space, gymnasium.spaces.Box):
        return gymnasium.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gymnasium.spaces.Discrete):
        return gymnasium.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gymnasium.spaces.Dict):
        return gymnasium.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError()


class ChunkingWrapper(gymnasium.Wrapper):
    """
    Enables observation histories and receding horizon control.

    Accumulates observations into obs_horizon size chunks. Starts by repeating the first obs.

    Executes act_exec_horizon actions in the environment.
    """

    def __init__(self, env: gymnasium.Env, obs_horizon: int, act_exec_horizon: Optional[int]):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon

        self.current_obs = deque(maxlen=self.obs_horizon)

        self.observation_space = space_stack(
            self.env.observation_space, self.obs_horizon
        )
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(
                self.env.action_space, self.act_exec_horizon
            )

    def step(self, action, *args):
        act_exec_horizon = self.act_exec_horizon
        if act_exec_horizon is None:
            action = [action]
            act_exec_horizon = 1

        assert len(action) >= act_exec_horizon

        for i in range(act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)
        return (stack_obs(self.current_obs), reward, done, trunc, info)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


def post_stack_obs(obs, obs_horizon=1):
    if obs_horizon != 1:
        # TODO: Support proper stacking
        raise NotImplementedError("Only obs_horizon=1 is supported for now")
    obs = {k: v[None] for k, v in obs.items()}
    return obs
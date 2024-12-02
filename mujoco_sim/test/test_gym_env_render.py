import time

import gym
import mujoco
import mujoco.viewer
import numpy as np

import mujoco_sim
from mujoco_sim.envs.wrappers import SpacemouseIntervention, CustomObsWrapper, ObsWrapper, GripperCloseEnv, XYZGripperCloseEnv, XYZQzGripperCloseEnv


env = gym.make("ur5ePegInHoleGymEnv-v0", render_mode="human")
env = XYZQzGripperCloseEnv(env)
env = SpacemouseIntervention(env)
env = CustomObsWrapper(env)
env = gym.wrappers.FlattenObservation(env)


action_spec = env.action_space
print(f"Action space: {action_spec}")

observation_spec = env.observation_space
print(f"Observation space: {observation_spec}")

def sample():
    # a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    a = np.zeros(action_spec.shape, dtype=action_spec.dtype)

    return a.astype(action_spec.dtype)


obs, info = env.reset()

for i in range(300):
    a = sample()
    obs, rew, done, truncated, info = env.step(a)

    if done:
        obs, info = env.reset()

# Properly close the environment
env.close()
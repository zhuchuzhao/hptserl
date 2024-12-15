import time

import gymnasium
import mujoco
import mujoco.viewer
import numpy as np
import mujoco_sim


# env = gymnasium.make("ur5ePegInHoleGymEnv_state-v0", render_mode="human")
env = gymnasium.make("ur5ePegInHoleGymEnv_vision-v0", render_mode="human")


action_spec = env.action_space
print(f"Action space: {action_spec}")

observation_spec = env.observation_space
print(f"Observation space: {observation_spec}")

def sample():
    # a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    a = np.zeros(action_spec.shape, dtype=action_spec.dtype)

    return a.astype(action_spec.dtype)


obs, info = env.reset()

for i in range(3000):
    a = sample()
    obs, rew, done, truncated, info = env.step(a)

    if done or truncated:
        obs, info = env.reset()

# Properly close the environment
env.close()
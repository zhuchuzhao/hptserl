import time
import mujoco
import mujoco.viewer
import numpy as np
import glfw
import gymnasium

from mujoco_sim import envs
from mujoco_sim.envs.wrappers import SpacemouseIntervention, ObsWrapper, GripperCloseEnv, XYZGripperCloseEnv, XYZQzGripperCloseEnv


# glfw init
glfw.init()

env = envs.ur5ePegInHoleGymEnv()
# env = XYZGripperCloseEnv(env)
# env = XYZQzGripperCloseEnv(env)
env = GripperCloseEnv(env)

# env = SpacemouseIntervention(env)
# env = CustomObsWrapper(env)

env = gymnasium.wrappers.FlattenObservation(env)

# Unwrapping the environment
unwrapped_env = env.unwrapped
m = unwrapped_env.model
d = unwrapped_env.data

action_spec = env.action_space
print(f"Action space: {action_spec}")

observation_spec = env.observation_space
print(f"Observation space: {observation_spec}")

def sample():
    # a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    a = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    print(f"Sampled action: {a}")
    return a.astype(action_spec.dtype)


reset = False
# KEY_SPACE = 32 # the key code for key ´space´
KEY_SPACE = 92 # the key code for key ´#´

action = sample()  # Generate an initial action sample
last_sample_time = time.time()  # Track the last sample time



def key_callback(keycode):
    # print(f"Key pressed: {keycode}")
    if keycode == KEY_SPACE:
        global reset
        reset = True


env.reset()
start_time = time.time()
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback, show_right_ui= False) as viewer:
    start = time.time()
    # env.external_viewer = viewer
    # env.reset()

    while viewer.is_running():
        if reset:
            env.reset()
            action = sample()  # Generate a new action sample
            last_sample_time = time.time()  # Reset the action timer
            reset = False
        else:
            step_start = time.time()

            # Update the action every 3 seconds
            if time.time() - last_sample_time >= 10.0:
                action = sample()  # Generate a new action sample
                last_sample_time = time.time()  # Update the last sample time

            env.step(action)
            viewer.sync()

            time_until_next_step = unwrapped_env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
viewer.close()
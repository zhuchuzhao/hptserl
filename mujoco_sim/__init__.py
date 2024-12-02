from mujoco_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gym.envs.registration import register

register(
    id="ur5ePegInHoleGymEnv-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    # max_episode_steps=1000,
)


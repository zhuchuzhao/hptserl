from mujoco_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gymnasium.envs.registration import register, WrapperSpec

register(
    id="ur5ePegInHoleGymEnv-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    additional_wrappers=(WrapperSpec(
    name='SpacemouseIntervention',
    entry_point='mujoco_sim.envs.wrappers:SpacemouseIntervention',
    kwargs={},
    ),
    ),
    kwargs={
        "config": {
            "UR5E_CONFIG": {
                "port_xy_randomize": True,
                "port_z_randomize": True,
            },
            },
    },
)
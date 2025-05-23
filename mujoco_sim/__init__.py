from mujoco_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
import numpy as np

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gymnasium.envs.registration import register, WrapperSpec

register(
    id="ur5ePegInHoleGymEnv_state-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    additional_wrappers=(
        WrapperSpec(
            name='GripperCloseEnv',
            entry_point='mujoco_sim.envs.wrappers:GripperCloseEnv',  # Replace with actual module path
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name='SpacemouseIntervention',
            entry_point='mujoco_sim.envs.wrappers:SpacemouseIntervention',
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name='ObsWrapper',
            entry_point='mujoco_sim.envs.wrappers:ObsWrapper',  # Replace with actual module path
            kwargs={
                # 'proprio_keys': [
                #     "ur5e/tcp_pose",
                #     "ur5e/wrist_force",
                #     "ur5e/wrist_torque",
                # ],
            },     
                        ),
    ),
    kwargs={
        "config": {
            "UR5E_CONFIG": {
            "restrict_cartesian_bounds": True,
            "port_xy_randomize": False,  # Randomize port xy placement
            "port_z_randomize": False,  # Randomize port z placement
            "port_orientation_randomize": False,  # Randomize port placement
            "max_port_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },   
            "tcp_xyz_randomize": False,  # Randomize tcp xyz placement
            "tcp_orient_randomize": True,
            "max_tcp_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },                
            },
            "ENV_CONFIG": {
                "image_obs": False,  # Use image observations
            },
        },
    },
)

register(
    id="ur5ePegInHoleGymEnv_vision-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    additional_wrappers=(
        # WrapperSpec(
        #     name='GripperCloseEnv',
        #     entry_point='mujoco_sim.envs.wrappers:GripperCloseEnv',  # Replace with actual module path
        #     kwargs={},  # Add any necessary kwargs for this wrapper
        # ),
        WrapperSpec(
            name='SpacemouseIntervention',
            entry_point='mujoco_sim.envs.wrappers:SpacemouseIntervention',
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name='ObsWrapper',
            entry_point='mujoco_sim.envs.wrappers:ObsWrapper',  # Replace with actual module path
            kwargs={
                'proprio_keys': [
                    "ur5e/tcp_pose",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                ],
            },     
                        ),
    ),
    kwargs={
        "config": {
            "UR5E_CONFIG": {
            "port_xy_randomize": False,  # Randomize port xy placement
            "port_z_randomize": False,  # Randomize port z placement
            "port_orientation_randomize": False,  # Randomize port placement
            "max_port_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },   
            "tcp_xyz_randomize": False,  # Randomize tcp xyz placement
            "tcp_orient_randomize": True,
            "max_tcp_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },                
            },
            "ENV_CONFIG": {
                "image_obs": True,  # Use image observations
            },
        },
    },
)



register(
    id="ur5ePegInHoleFixedGymEnv_state-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleFixedGymEnv",
    additional_wrappers=(
        WrapperSpec(
            name='XYZGripperCloseEnv',
            entry_point='mujoco_sim.envs.wrappers:XYZGripperCloseEnv',  # Replace with actual module path
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name='SpacemouseIntervention',
            entry_point='mujoco_sim.envs.wrappers:SpacemouseIntervention',
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name='ObsWrapper',
            entry_point='mujoco_sim.envs.wrappers:ObsWrapper',  # Replace with actual module path
            kwargs={
                'proprio_keys': [
                    "controller_pose",
                    "ur5e/tcp_pose",
                    "ur5e/tcp_vel",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                    "connector_pose",
                ],
            },     
                        ),
            WrapperSpec(
            name='ChunkingWrapper',
            entry_point='mujoco_sim.envs.wrappers:ChunkingWrapper',  # Replace with actual module path
            kwargs={
            'obs_horizon': 1,
            'act_exec_horizon': None,
            },     
                        ),
    ),
    kwargs={
        "config": {
            "UR5E_CONFIG": {
            "restrict_cartesian_bounds": True,
            "port_xy_randomize": False,  # Randomize port xy placement
            "port_z_randomize": False,  # Randomize port z placement
            "port_orientation_randomize": False,  # Randomize port placement
            "max_port_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },   
            "tcp_xyz_randomize": True,  # Randomize tcp xyz placement
            "tcp_randomization_bounds": np.array([[-0.005, -0.005, 0.06], [0.005, 0.005, 0.06]]),  # Randomization bounds for port positions xyz
            "tcp_orient_randomize": False,
            "max_tcp_orient_randomize": {
                "x": 45,  # Maximum deviation in degrees around x-axis
                "y": 45,  # Maximum deviation in degrees around y-axis
                "z": 45,  # Maximum deviation in degrees around z-axis
            },                
            },
            "REWARD_CONFIG" : {
            "reward_shaping": True,  # Use dense reward shaping
            },
            "ENV_CONFIG": {
                "image_obs": False,  # Use image observations
            },
        },
    },
)





register(
    id="ur5ePegInHoleFixedGymEnv_vision-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleFixedGymEnv",
    additional_wrappers=(
        WrapperSpec(
            name='XYZGripperCloseEnv',
            entry_point='mujoco_sim.envs.wrappers:XYZGripperCloseEnv',  # Replace with actual module path
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name='SpacemouseIntervention',
            entry_point='mujoco_sim.envs.wrappers:SpacemouseIntervention',
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name='ObsWrapper',
            entry_point='mujoco_sim.envs.wrappers:ObsWrapper',  # Replace with actual module path
            kwargs={
                'proprio_keys': [
                    "ur5e/tcp_pose",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                ],
            },     
                        ),
            WrapperSpec(
            name='ChunkingWrapper',
            entry_point='mujoco_sim.envs.wrappers:ChunkingWrapper',  # Replace with actual module path
            kwargs={
            'obs_horizon': 1,
            'act_exec_horizon': None
            },     
                        ),
    ),
    kwargs={
        "config": {
            "UR5E_CONFIG": {
            "port_xy_randomize": False,  # Randomize port xy placement
            "port_z_randomize": False,  # Randomize port z placement
            "port_orientation_randomize": False,  # Randomize port placement
            "max_port_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },   
            "tcp_xyz_randomize": False,  # Randomize tcp xyz placement
            "tcp_orient_randomize": True,
            "max_tcp_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },                
            },
            "ENV_CONFIG": {
                "image_obs": True,  # Use image observations
            },
        },
    },
)



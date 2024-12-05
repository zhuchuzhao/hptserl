from mujoco_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gymnasium.envs.registration import register, WrapperSpec

register(
    id="ur5ePegInHoleGymEnv_easy-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
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
            name='CustomObsWrapper',
            entry_point='mujoco_sim.envs.wrappers:CustomObsWrapper',  # Replace with actual module path
            kwargs={
                        },        
                        ),
        WrapperSpec(
            name='FlattenObservation',
            entry_point='gymnasium.wrappers:FlattenObservation',
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
    ),
    kwargs={
        "config": {
            "UR5E_CONFIG": {
                "tcp_xyz_randomize": True,  
                "mocap_orient": True,
            },
            },
    },
)

register(
    id="ur5ePegInHoleGymEnv_medium-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    additional_wrappers=(
        WrapperSpec(
            name='XYZQzGripperCloseEnv',
            entry_point='mujoco_sim.envs.wrappers:XYZQzGripperCloseEnv',  # Replace with actual module path
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name='SpacemouseIntervention',
            entry_point='mujoco_sim.envs.wrappers:SpacemouseIntervention',
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name='CustomObsWrapper',
            entry_point='mujoco_sim.envs.wrappers:CustomObsWrapper',  # Replace with actual module path
            kwargs={
                'keys_to_keep': [
                    "ur5e/tcp_pose",
                    "ur5e/tcp_vel",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                    "connector_pose",
                ],
            },     
                        ),
        WrapperSpec(
            name='FlattenObservation',
            entry_point='gymnasium.wrappers:FlattenObservation',
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
    ),
    kwargs={
        "config": {
            "UR5E_CONFIG": {
            "port_xy_randomize": True,  # Randomize port xy placement
            "port_z_randomize": True,  # Randomize port z placement
            "port_orientation_randomize": True,  # Randomize port placement
            "max_port_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 180,  # Maximum deviation in degrees around z-axis
            },   
            "tcp_xyz_randomize": True,  # Randomize tcp xyz placement
            "mocap_orient": False,
            "max_mocap_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 180,  # Maximum deviation in degrees around z-axis
            },            
            },
            },
    },
)

register(
    id="ur5ePegInHoleGymEnv_hard-v0",
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
            name='CustomObsWrapper',
            entry_point='mujoco_sim.envs.wrappers:CustomObsWrapper',  # Replace with actual module path
            kwargs={
                'keys_to_keep': [
                    "ur5e/tcp_pose",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                ],
            },     
                        ),
        WrapperSpec(
            name='FlattenObservation',
            entry_point='gymnasium.wrappers:FlattenObservation',
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
    ),
    kwargs={
        "config": {
            "UR5E_CONFIG": {
            "port_xy_randomize": True,  # Randomize port xy placement
            "port_z_randomize": True,  # Randomize port z placement
            "port_orientation_randomize": True,  # Randomize port placement
            "max_port_orient_randomize": {
                "x": 30,  # Maximum deviation in degrees around x-axis
                "y": 30,  # Maximum deviation in degrees around y-axis
                "z": 180,  # Maximum deviation in degrees around z-axis
            },   
            "tcp_xyz_randomize": True,  # Randomize tcp xyz placement
            "mocap_orient": False,
            "max_mocap_orient_randomize": {
                "x": 30,  # Maximum deviation in degrees around x-axis
                "y": 30,  # Maximum deviation in degrees around y-axis
                "z": 180,  # Maximum deviation in degrees around z-axis
            },                
            },
            },
    },
)

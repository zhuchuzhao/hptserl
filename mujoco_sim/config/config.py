import numpy as np
import mujoco


class PegEnvConfig():
    """Set the configuration for FrankaEnv."""
    def __init__(self, **kwargs):
        # General Environment Configuration
        self.ENV_CONFIG = {
            "action_scale": np.array([1, 1, 1]),  # Scaling factors for position, orientation, and gripper control
            "control_dt": 0.02,  # Time step for controller updates
            "physics_dt": 0.002,  # Time step for physics simulation
            "time_limit": 20.0,  # Time limit for each episode
            "seed": 0,  # Random seed
            "image_obs": False,  # Use image observations
        }

        self.DEFAULT_CAM_CONFIG = {
        'type': mujoco.mjtCamera.mjCAMERA_FREE,  # Camera type
        'fixedcamid': 0,                            # ID of the fixed camera
        'lookat': np.array([-0.13658866, -0.07953108, -0.12050411]),        # Point the camera looks at
        'distance': 0.7681,                            # Distance from the lookat point
        'azimuth': -170.1003,                           # Horizontal angle
        'elevation': -19.9855,                         # Vertical angle
        }

        # UR5e Robot Configuration
        self.UR5E_CONFIG = {
            "home_position": np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]),  # Home joint angles
            "reset_position": np.array([-1.97248201, -1.80736221,  2.08230528, -1.84573939, -1.57079633,  1.16911064]),  # Reset joint angles
            "default_cartesian_bounds": np.array([[0.2, -0.3, 0.0], [0.5, 0.3, 0.5]]),  # Workspace boundaries in Cartesian space
            "restrict_cartesian_bounds": False,  # Whether to restrict the end effector to the Cartesian bounds
            "default_port_pos": np.array([0.4, 0.0, 0.0]),  # Default port position
            "port_sampling_bounds": np.array([[0.395, -0.05, 0], [0.405, 0.05, 0.1]]),  # Sampling range for port placement
            "connector_sampling_bounds": np.asarray([[0.3, -0.1], [0.4, 0.1]]),
            "port_xy_randomize": False,  # Randomize port xy placement
            "port_z_randomize": False,  # Randomize port z placement
            "port_orientation_randomize": False,  # Randomize port placement
            "max_port_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },   
            "tcp_xyz_randomize": False,  # Randomize tcp xyz placement
            "tcp_orient_randomize": True,  # Orient the tcp to the port
            "max_tcp_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },
            "tcp_randomization_bounds": np.array([[-0.005, -0.005, 0.06], [0.005, 0.005, 0.06]]),  # Randomization bounds for port positions xyz
            "reset_tolerance": 0.06,
        }

        # Controller Configuration
        self.CONTROLLER_CONFIG = {
            "trans_damping_ratio": 0.996,  # Damping ratio for translational control
            "rot_damping_ratio": 0.286,  # Damping ratio for rotational control
            "error_tolerance_pos": 0.00025,  # Position error tolerance
            "error_tolerance_ori": 0.0005,  # Orientation error tolerance
            "trans_clip_min": np.array([-0.01, -0.01, -0.01]),  # Translational negative clipping limits
            "trans_clip_max": np.array([0.01, 0.01, 0.01]),  # Translational positive clipping limits
            "rot_clip_min": np.array([-0.03, -0.03, -0.03]),  # Rotational negative clipping limits
            "rot_clip_max": np.array([0.03, 0.03, 0.03]),  # Rotational positive clipping
            "method": "dynamics",  # Control method ("dynamics", "pinv", "svd", etc.)
            "inertia_compensation": False,  # Whether to compensate for inertia
            "pos_gains": (100, 100, 100),  # Proportional gains for position control
            # "ori_gains": (12.5, 12.5, 12.5),  # Proportional gains for orientation control
            "max_angvel": 4,  # Maximum angular velocity
            "integration_dt": 0.2,  # Integration time step for controller
            "gravity_compensation": True,  # Whether to compensate for gravity  
        }

        # Rendering Configuration
        self.RENDERING_CONFIG = {
            "width": 640,  # Rendering width
            "height": 480,  # Rendering height
        }

        # Reward Shaping
        self.REWARD_CONFIG = {
            "reward_shaping": False,  # Use dense reward shaping
            "dense_reward_weights": {
                "box_target": 1.0,  # Weight for reaching target position
            },
            "sparse_reward_weights": 1,  # Reward for completing the task
            "task_complete_tolerance": 0.003,  # Distance threshold for task completion
        }

        # Update configurations with provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                attr_value = getattr(self, key)
                if isinstance(attr_value, dict) and isinstance(value, dict):
                    attr_value.update(value)
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
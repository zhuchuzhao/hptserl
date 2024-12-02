import numpy as np


class PegEnvConfig():
    """Set the configuration for FrankaEnv."""
    # General Environment Configuration
    ENV_CONFIG = {
        "action_scale": np.array([0.005,0.005, 1]),  # Scaling factors for position, orientation, and gripper control
        "control_dt": 0.02,  # Time step for controller updates
        "physics_dt": 0.002,  # Time step for physics simulation
        "time_limit": 20.0,  # Time limit for each episode
        "seed": 0,  # Random seed
    }

    # UR5e Robot Configuration
    UR5E_CONFIG = {
        "home_position": np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]),  # Home joint angles
        "reset_position": np.array([-1.43822003, -1.95706484,  1.53889027, -1.15262176, -1.57079633,  1.70337263]),  # Reset joint angles
        "default_cartesian_bounds": np.array([[0.1, -0.3, 0.0], [0.4, 0.3, 0.5]]),  # Workspace boundaries in Cartesian space
        "restrict_cartesian_bounds": True,  # Whether to restrict the end effector to the Cartesian bounds
        "default_port_pos": np.array([0.4, 0.0, 0.0]),  # Default port position
        "port_sampling_bounds": np.array([[0.395, -0.05, 0], [0.405, 0.05, 0.1]]),  # Sampling range for port placement
        #TODO: 1.no port randomization 2. randomize xy 3.randomize all 6 dof(limited)
        "port_xy_randomize": True,  # Randomize port xy placement
        "port_z_randomize": False,  # Randomize port z placement
        "port_orientation_randomize": True,  # Randomize port placement
        "max_port_orient": 30,  # Maximum orientation deviation for port placement
        "tcp_xyz_randomize": True,  # Randomize tcp xyz placement
        "mocap_orient": True,  # Orient the tcp to the port
        "randomization_bounds": np.array([[-0.025, -0.025, 0.05], [0.025, 0.025, 0.06]]),  # Randomization bounds for port positions
        "reset_tolerance": 0.002,
    }

    # Controller Configuration
    CONTROLLER_CONFIG = {
        "trans_damping_ratio": 0.996,  # Damping ratio for translational control
        "rot_damping_ratio": 0.286,  # Damping ratio for rotational control
        "error_tolerance_pos": 0.001,  # Position error tolerance
        "error_tolerance_ori": 0.001,  # Orientation error tolerance
        "max_pos_error": 0.01,  # Maximum position error
        "max_ori_error": 0.03,  # Maximum orientation error
        "method": "dynamics",  # Control method ("dynamics", "pinv", "svd", etc.)
        "inertia_compensation": False,  # Whether to compensate for inertia
        "pos_gains": (100, 100, 100),  # Proportional gains for position control
        # "ori_gains": (12.5, 12.5, 12.5),  # Proportional gains for orientation control
        "max_angvel": 4,  # Maximum angular velocity
        "integration_dt": 0.2,  # Integration time step for controller
        "gravity_compensation": True,  # Whether to compensate for gravity  
    }

    # Rendering Configuration
    RENDERING_CONFIG = {
        "width": 640,  # Rendering width
        "height": 480,  # Rendering height
    }

    # Reward Shaping
    REWARD_CONFIG = {
        "reward_shaping": True,  # Use dense reward shaping
        "dense_reward_weights": {
            "box_target": 8.0,  # Weight for reaching target position
            "gripper_box": 4.0,  # Weight for gripper being close to connector
            "no_floor_collision": 0.25,  # Penalty for floor collisions
            "grasping_reward": 0.25,  # Reward for successful grasp
        },
        "sparse_reward_weights": 12.5,  # Reward for completing the task
        "task_complete_tolerance": 0.002,  # Distance threshold for task completion
    }
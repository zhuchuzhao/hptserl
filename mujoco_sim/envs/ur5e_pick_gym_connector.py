from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gym
import mujoco
import numpy as np
from gym import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from mujoco_sim.controllers import Controller
from mujoco_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from mujoco_sim.config import PegEnvConfig

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "ur5e_arena.xml"

class ur5ePegInHoleGymEnv(MujocoGymEnv):
    """UR5e peg-in-hole environment in Mujoco."""
    def __init__(self, render_mode: Literal["rgb_array", "human"] = "rgb_array",):
        config = PegEnvConfig()

        # Rendering configuration
        self.render_width = config.RENDERING_CONFIG["width"]
        self.render_height = config.RENDERING_CONFIG["height"]
        self.camera_id = (0, 1)  # Camera IDs for rendering
        render_spec = GymRenderingSpec(
        height=config.RENDERING_CONFIG["height"],
        width=config.RENDERING_CONFIG["width"],
        )

        super().__init__(
            xml_path=_XML_PATH,
            control_dt=config.ENV_CONFIG["control_dt"],
            physics_dt=config.ENV_CONFIG["physics_dt"],
            time_limit=config.ENV_CONFIG["time_limit"],
            seed=config.ENV_CONFIG["seed"],
            render_spec=render_spec,
        )

        self._action_scale = config.ENV_CONFIG["action_scale"]
        self.render_mode = render_mode

        # UR5e-specific configuration
        self.ur5e_home = config.UR5E_CONFIG["home_position"]
        self.ur5e_reset = config.UR5E_CONFIG["reset_position"]
        self.cartesian_bounds = config.UR5E_CONFIG["default_cartesian_bounds"]
        self.restrict_cartesian_bounds = config.UR5E_CONFIG["restrict_cartesian_bounds"]
        self.default_port_pos = config.UR5E_CONFIG["default_port_pos"]
        self.port_sampling_bounds = config.UR5E_CONFIG["port_sampling_bounds"]
        self.tcp_xyz_randomize = config.UR5E_CONFIG["tcp_xyz_randomize"]
        self.port_xy_randomize = config.UR5E_CONFIG["port_xy_randomize"]
        self.port_z_randomize = config.UR5E_CONFIG["port_z_randomize"]
        self.port_orientation_randomize = config.UR5E_CONFIG["port_orientation_randomize"]
        self.max_port_orient = config.UR5E_CONFIG["max_port_orient"]
        self.mocap_orient = config.UR5E_CONFIG["mocap_orient"]
        self.randomization_bounds = config.UR5E_CONFIG["randomization_bounds"]
        self.reset_tolerance = config.UR5E_CONFIG["reset_tolerance"]

        # Reward configuration
        self.reward_config = config.REWARD_CONFIG 

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.external_viewer = None

        # Caching UR5e joint and actuator IDs.
        self._ur5e_dof_ids = np.asarray([
            self._model.joint("shoulder_pan_joint").id,   # Joint 1: Shoulder pan
            self._model.joint("shoulder_lift_joint").id,  # Joint 2: Shoulder lift
            self._model.joint("elbow_joint").id,          # Joint 3: Elbow
            self._model.joint("wrist_1_joint").id,        # Joint 4: Wrist 1
            self._model.joint("wrist_2_joint").id,        # Joint 5: Wrist 2
            self._model.joint("wrist_3_joint").id         # Joint 6: Wrist 3
        ])
        
        self._ur5e_ctrl_ids = np.asarray([
            self._model.actuator("shoulder_pan").id,      # Actuator for Joint 1
            self._model.actuator("shoulder_lift").id,     # Actuator for Joint 2
            self._model.actuator("elbow").id,             # Actuator for Joint 3
            self._model.actuator("wrist_1").id,           # Actuator for Joint 4
            self._model.actuator("wrist_2").id,           # Actuator for Joint 5
            self._model.actuator("wrist_3").id            # Actuator for Joint 6
        ])
        
        self._gripper_ctrl_id = self._model.actuator("hande_fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._port_id = self._model.body("port_adapter").id
        self._port1_id = self._model.body("port1").id
        self._port_site = self._model.site("port_top").id
        self._port_site_id = self._model.site("port_top").id
        
        # Updated identifiers for the geometries and sensors
        self._floor_geom = self._model.geom("floor").id
        self._left_finger_geom = self._model.geom("left_pad1").id
        self._right_finger_geom = self._model.geom("right_pad1").id
        self._hand_geom = self._model.geom("hande_base").id
        self._connector_head_geom = self._model.geom("connector_head").id

        # Update the default cartesian bounds geom size and position
        self._cartesian_bounds_geom_id = self._model.geom("cartesian_bounds").id
        # Extract bounds
        lower_bounds, upper_bounds = self.cartesian_bounds
        center = (upper_bounds + lower_bounds) / 2.0
        size = (upper_bounds - lower_bounds) / 2.0
        self._model.geom_size[self._cartesian_bounds_geom_id] = size
        self._model.geom_pos[self._cartesian_bounds_geom_id] = center

        #preallocate memory
        self.quat_err = np.zeros(4)
        self.quat_conj = np.zeros(4)
        self.ori_err = np.zeros(3)

        self.controller = Controller(
        model=self._model,
        data=self._data,
        site_id=self._pinch_site_id,
        dof_ids=self._ur5e_dof_ids,
        config=config.CONTROLLER_CONFIG,
        )
        obs_bound = 1e6
        #TODO: 1.max obs space (everything except gripper) 2.wrist force, wrist torque, tcp vel 
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "ur5e/tcp_pose": spaces.Box(
                            -obs_bound, obs_bound, shape=(6,), dtype=np.float32
                        ),
                        "ur5e/tcp_vel": spaces.Box(
                            -obs_bound, obs_bound, shape=(3,), dtype=np.float32
                        ),
                        "ur5e/gripper_pos": spaces.Box(
                            -obs_bound, obs_bound, shape=(1,), dtype=np.float32
                        ),
                        "ur5e/joint_pos": spaces.Box(-obs_bound, obs_bound, shape=(6,), dtype=np.float32),
                        "ur5e/joint_vel": spaces.Box(-obs_bound, obs_bound, shape=(6,), dtype=np.float32),
                        "ur5e/wrist_force": spaces.Box(-obs_bound, obs_bound, shape=(3,), dtype=np.float32),
                        "ur5e/wrist_torque": spaces.Box(-obs_bound, obs_bound, shape=(3,), dtype=np.float32),
                        "connector_pose": spaces.Box(
                            -obs_bound, obs_bound, shape=(6,), dtype=np.float32
                        ),
                    }
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self._viewer = MujocoRenderer(
            self.model,
            self.data,
            width= render_spec.width, height=render_spec.height,
        )
        self._viewer.render(self.render_mode)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self._ur5e_dof_ids] = self.ur5e_reset
        self._data.qvel[self._ur5e_dof_ids] = 0  # Ensure joint velocities are zero
        mujoco.mj_forward(self._model, self._data)

        # Define plate bounds
        # Randomize port position if flag is True
        # Set default port position
        port_xy = self.default_port_pos[:2]
        port_z = self.default_port_pos[2]

        # Randomize x and y position if flag is true
        if self.port_xy_randomize:
            port_xy = np.random.uniform(self.port_sampling_bounds[0][:2], self.port_sampling_bounds[1][:2])
        self._model.body_pos[self._port_id][:2] = port_xy

        # Randomize z position if flag is true
        if self.port_z_randomize:
            port_z = np.random.uniform(self.port_sampling_bounds[0][2], self.port_sampling_bounds[1][2])
        self._model.body_pos[self._port_id][2] = port_z

        # Set port orientation
        if self.port_orientation_randomize:
            max_angle_rad = np.deg2rad(self.max_port_orient)  # Limit to ±45 degrees
            random_angles = np.random.uniform(-max_angle_rad, max_angle_rad, size=3)
            quat_des = np.zeros(4)
            mujoco.mju_euler2Quat(quat_des, random_angles, "xyz")
            self._model.body_quat[self._port_id] = quat_des

        mujoco.mj_forward(self._model, self._data)

        plate_pos = self._data.geom("plate").xpos
        half_width, half_height, half_depth = self._model.geom("plate").size 
        local_vertices = np.array([
            [ half_width,  half_height,  half_depth],
            [ half_width,  half_height, -half_depth],
            [ half_width, -half_height,  half_depth],
            [ half_width, -half_height, -half_depth],
            [-half_width,  half_height,  half_depth],
            [-half_width,  half_height, -half_depth],
            [-half_width, -half_height,  half_depth],
            [-half_width, -half_height, -half_depth],
        ])
        # rotation_matrix = self.data.site_xmat[self._port_site_id].reshape(3,3)
        rotation_matrix = self.data.xmat[self._port_id].reshape(3, 3)
        rotated_vertices = local_vertices @ rotation_matrix.T + plate_pos
        # Find the lowest z-coordinate among the rotated vertices
        z_coords = rotated_vertices[:, 2]
        z_min = np.min(z_coords)
        
        if z_min < 0.0:
            z_offset = -z_min
            print(f"Adjusting port height by {z_offset} m.")
        else:
            z_offset = 0.0

        self._model.body_pos[self._port_id][2] += z_offset
        mujoco.mj_forward(self._model, self._data)

        # Update the geom size and position
        if self.restrict_cartesian_bounds:
            # Update the cartesian bounds
            self._model.geom_size[self._cartesian_bounds_geom_id] = self._model.geom("plate").size * np.array([0.5, 0.6, 1]) + np.array([0.008, 0, 0.075])
            self._model.geom_pos[self._cartesian_bounds_geom_id] = self._data.geom("plate").xpos + np.array([0, 0, 0.075] @ rotation_matrix.T)
            self._model.geom_quat[self._cartesian_bounds_geom_id] = self._model.body_quat[self._port_id]

        port_xyz = self.data.site_xpos[self._port_site_id]
        if self.tcp_xyz_randomize:
            # Generate random XYZ offsets in the local frame
            # rotation_matrix = self.data.xmat[self._port_id].reshape(3, 3)
            random_xyz_local = np.random.uniform(*self.randomization_bounds) 
            random_xyz_global = random_xyz_local @ rotation_matrix.T + port_xyz

            # Independent variations for x, y, and z axes
            # Sample mocap position with independent variations
            self._data.mocap_pos[0] = np.array([
            random_xyz_global[0],
            random_xyz_global[1],
            random_xyz_global[2] 
            ] )
        else:
            # Fixed addition for z axis only
            self._data.mocap_pos[0] = np.array([
            port_xyz[0],
            port_xyz[1],
            port_xyz[2] + 0.1
            ])


        if self.mocap_orient:
            quat_des = np.zeros(4)
            mujoco.mju_mat2Quat(quat_des, self.data.site_xmat[self._port_site_id])
            self._data.mocap_quat[0] = quat_des
        mujoco.mj_forward(self._model, self._data)

        step_count = 0
        while step_count < 1000:
            current_tcp_pos = self._data.sensor("hande/pinch_pos").data.copy()
            distance = np.linalg.norm(self._data.mocap_pos[0] - current_tcp_pos)
            if distance <= self.reset_tolerance:
                break  # Goal reached
            self.step()
            step_count += 1
        
        print(f"Resetting environment after {step_count} steps.")

        # plate_pos = self._data.geom("plate").xpos
        # plate_size = self._model.geom("plate").size
        # connector_radius = self._model.geom("connector_back").size # Assuming the first size is the radius

        # plate_bounds = [
        #     [plate_pos[0] - plate_size[0] - connector_radius[0], plate_pos[1] - plate_size[1] - connector_radius[0]],
        #     [plate_pos[0] + plate_size[0] + connector_radius[1], plate_pos[1] + plate_size[1] + connector_radius[1]],
        # ]

        # # Sample connector position avoiding the plate bounds
        # while True:
        #     connector_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        #     if not (
        #         plate_bounds[0][0] <= connector_xy[0] <= plate_bounds[1][0]
        #         and plate_bounds[0][1] <= connector_xy[1] <= plate_bounds[1][1]
        #     ):
        #         break
        # self._data.jnt("connector").qpos[:3] = (*connector_xy, self._connector_z)

        # # Reset mocap body to home position.
        # self._data.mocap_pos[0] = (*connector_xy + 0.01, self._connector_z + 0.025)

        # Cache the initial block height.
        self._z_init = self._data.sensor("connector_head_pos").data[2]        
        obs = self._compute_observation()
        return obs, {}
    
    def get_state(self) -> np.ndarray:
        """Return MjSimState instance for current state."""
        return np.concatenate([[self.data.time], np.copy(self.data.qpos), np.copy(self.data.qvel)], axis=0)

    def step(
        self, action: np.ndarray= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        delta_x, delta_y, delta_z, delta_qx, delta_qy, delta_qz, grasp = action

        #TODO: vectorized env. 1.for cpu 2. for gpu using using mjx
        #TODO: Add 3 action space 1.delta x,y,z 2. delta x,y,z, delta qz 3. all deltas
        #TODO: add delta instead of q
        #TODO: visualize action space

        # Set the position.
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([delta_x, delta_y, delta_z]) * self._action_scale[0]
        # Transform to OBB's local frame
        cartesian_pos = self._data.geom("cartesian_bounds").xpos
        cartesian_half_extents = self._model.geom("cartesian_bounds").size 
        obb_rotation = self.data.xmat[self._port_id].reshape(3, 3)
        local_pos = (pos + dpos - cartesian_pos) @ obb_rotation.T
        clipped_local_pos = np.clip(local_pos, -cartesian_half_extents, cartesian_half_extents)
        new_pos = clipped_local_pos @ obb_rotation + cartesian_pos
        self._data.mocap_pos[0] = new_pos
        # npos = np.clip(pos + dpos, *self.cartesian_bounds)
        # self._data.mocap_pos[0] = npos

        # Set the orientation
        ori = self._data.mocap_quat[0].copy()

        # If any orientation deltas are provided, update orientation
        if any([delta_qx, delta_qy, delta_qz]):
            new_ori = np.zeros(4)
            quat_des = np.zeros(4)
            nori = np.asarray([delta_qx, delta_qy, delta_qz], dtype=np.float32) * self._action_scale[1]
            mujoco.mju_euler2Quat(new_ori, nori, 'xyz')
            mujoco.mju_mulQuat(quat_des, new_ori, ori)
            self._data.mocap_quat[0] = quat_des

        # Set gripper grasp if provided
        if grasp != 0.0:
            g = self._data.ctrl[self._gripper_ctrl_id] / 255
            dg = grasp * self._action_scale[2]
            ng = np.clip(g + dg, 0.0, 1.0)
            self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):

            ctrl = self.controller.control(
                pos=self._data.mocap_pos[0].copy(),
                ori=self._data.mocap_quat[0].copy(),
            )  
            # Set the control signal.
            self._data.ctrl[self._ur5e_ctrl_ids] = ctrl
            mujoco.mj_step(self._model, self._data)
            
        obs = self._compute_observation()
        # print(self._data.qpos[self._ur5e_dof_ids])

        rew, task_complete = self._compute_reward()
        terminated = self.time_limit_exceeded() or task_complete

        return obs, rew, terminated, False, {"succeed": task_complete}

    def render(self) -> np.ndarray:
        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.camera = cam_id  # Set the camera based on cam_id
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array")
            )
        return rendered_frames
    
    def _get_contact_info(self, geom1_id: int, geom2_id: int) -> bool:
        """Get distance and normal for contact between geom1 and geom2."""
        # Iterate through contacts
        for contact in self._data.contact[:self._data.ncon]:
            # Check if contact involves geom1 and geom2
            if {contact.geom1, contact.geom2} == {geom1_id, geom2_id}:
                distance = contact.dist
                normal = contact.frame[:3]
                return True, distance, normal
        return False, float('inf'), np.zeros(3)  # No contact

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("hande/pinch_pos").data.astype(np.float32)
        tcp_ori_quat = self._data.sensor("hande/pinch_quat").data.astype(np.float32)
        tcp_ori_euler = np.zeros(3)
        mujoco.mju_quat2Vel(tcp_ori_euler, tcp_ori_quat, 1.0)
        obs["state"]["ur5e/tcp_pose"] = np.concatenate((tcp_pos,tcp_ori_euler)).astype(np.float32)                     

        tcp_vel = self._data.sensor("hande/pinch_vel").data
        obs["state"]["ur5e/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            [self._data.ctrl[self._gripper_ctrl_id] / 255], dtype=np.float32
        )
        obs["state"]["ur5e/gripper_pos"] = gripper_pos

        joint_pos = np.stack(
            [self._data.sensor(f"ur5e/joint{i}_pos").data for i in range(1, 7)],
        ).ravel()
        obs["state"]["ur5e/joint_pos"] = joint_pos.astype(np.float32)

        joint_vel = np.stack(
            [self._data.sensor(f"ur5e/joint{i}_vel").data for i in range(1, 7)],
        ).ravel()
        obs["state"]["ur5e/joint_vel"] = joint_vel.astype(np.float32)

        wrist_force = self._data.sensor("ur5e/wrist_force").data
        obs["state"]["ur5e/wrist_force"] = wrist_force.astype(np.float32)

        wrist_torque = self._data.sensor("ur5e/wrist_torque").data
        obs["state"]["ur5e/wrist_torque"] = wrist_torque.astype(np.float32)

        connector_pos = self._data.sensor("connector_head_pos").data.astype(np.float32)
        connector_ori_quat = self._data.sensor("connector_head_quat").data.astype(np.float32)
        connector_ori_euler = np.zeros(3)
        mujoco.mju_quat2Vel(connector_ori_euler, connector_ori_quat, 1.0)
        obs["state"]["connector_pose"] = np.concatenate((connector_pos, connector_ori_euler)).astype(np.float32)

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs

    def _compute_reward(self) -> float:
        sensor_data = self._data.sensor
        connector_head_pos = sensor_data("connector_head_pos").data
        connector_head_ori = sensor_data("connector_head_quat").data
        tcp_pos = sensor_data("hande/pinch_pos").data
        connector_bottom_pos = sensor_data("connector_bottom_pos").data
        port_bottom_pos = sensor_data("port_bottom_pos").data
        port_bottom_quat = sensor_data("port_bottom_quat").data
        distance = np.linalg.norm(connector_bottom_pos - port_bottom_pos)
        z_distance = abs(connector_bottom_pos[2] - port_bottom_pos[2])

        # Orientation Control
        mujoco.mju_negQuat(self.quat_conj, connector_head_ori)
        mujoco.mju_mulQuat(self.quat_err, port_bottom_quat, self.quat_conj)
        mujoco.mju_quat2Vel(self.ori_err, self.quat_err, 1.0)
        distance += 0.5*np.linalg.norm(self.ori_err)

        # Task completion
        task_complete = distance < self.reward_config["task_complete_tolerance"]
        task_complete_z = z_distance < self.reward_config["task_complete_tolerance"]

        # Compute z distance for sparse reward
        if not self.reward_config["reward_shaping"]:
            return (self.reward_config["sparse_reward_weights"] if task_complete_z else 0.0), task_complete
        
        # Dense rewards with shaping
        dense_weights = self.reward_config["dense_reward_weights"]

        #TODO: change sparse reward to simple z distance 2. dense reward without tanh

        reward_components = {
        "box_target": lambda: max(1 - distance, 0),
        }

        # Combine only the active rewards
        reward = sum(
            dense_weights[component] * reward_components[component]()
            for component in dense_weights if component in reward_components
        )
            
        return reward, task_complete

if __name__ == "__main__":
    env = ur5ePegInHoleGymEnv()
    env.reset()
    for i in range(1000):
        env.step(np.random.uniform(-1, 1, 7))
        env.render()
    env.close()

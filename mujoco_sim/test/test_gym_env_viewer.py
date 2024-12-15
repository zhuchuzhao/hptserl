import time
import mujoco
from mujoco_sim.viewer.mujoco_viewer import MujocoViewer
import numpy as np
import gymnasium
from mujoco_sim import envs
from mujoco_sim.utils.viz import SliderController
from mujoco_sim.envs.wrappers import SpacemouseIntervention, CustomObsWrapper, ObsWrapper, GripperCloseEnv, XYZGripperCloseEnv, XYZQzGripperCloseEnv

# Initialize the environment and controller
env = envs.ur5ePegInHoleGymEnv()
action_spec = env.action_space

# Define indices for UR5e DOF and gripper
ur5e_dof_indices = env._ur5e_dof_ids
gripper_dof_index = env._gripper_ctrl_id
env = GripperCloseEnv(env)
env = SpacemouseIntervention(env)
env = CustomObsWrapper(env)
env = gymnasium.wrappers.FlattenObservation(env)

# Unwrapping the environment
unwrapped_env = env.unwrapped

controller = unwrapped_env.controller
# slider_controller = SliderController(controller)

# Sample a random action within the action space
def sample():
    a = np.zeros(action_spec.shape, dtype=action_spec.dtype)

    return a

# Environment data and variables
m = unwrapped_env.model
d = unwrapped_env.data
reset = False
KEY_SPACE = 32
action = sample()
last_sample_time = time.time()

# Reset the environment
env.reset()
viewer = MujocoViewer(m, d, hide_menus=True)


# Set up graph lines for UR5e DOF and gripper
for joint_idx in ur5e_dof_indices:
    viewer.add_line_to_fig(line_name=f"qpos_ur5e_joint_{joint_idx}", fig_idx=0)
viewer.add_line_to_fig(line_name="qpos_gripper", fig_idx=0)

# Set up figure properties for visualization
fig0 = viewer.figs[0]
fig0.title = "UR5e Joint Positions"
fig0.xlabel = "Timesteps"
fig0.flg_legend = True
fig0.figurergba[0] = 0.2
fig0.figurergba[3] = 0.2
fig0.gridsize[0] = 10
fig0.gridsize[1] = 5

# # Set up distinct colors for each axis in the target and current position lines
# viewer.add_line_to_fig(line_name="target_x", fig_idx=1, color=[0.85, 0.3, 0])  # Dark Orange
# viewer.add_line_to_fig(line_name="target_y", fig_idx=1, color=[0.2, 0.6, 0.2])  # Dark Green
# viewer.add_line_to_fig(line_name="target_z", fig_idx=1, color=[0.2, 0.4, 0.8])  # Deep Blue

# viewer.add_line_to_fig(line_name="current_x", fig_idx=1, color=[1, 0.5, 0.2])  # Light Orange
# viewer.add_line_to_fig(line_name="current_y", fig_idx=1, color=[0.4, 0.9, 0.4])  # Light Green
# viewer.add_line_to_fig(line_name="current_z", fig_idx=1, color=[0.3, 0.6, 1])  # Lighter Blue

# fig1 = viewer.figs[1]
# fig1.title = "End-Effector Position Tracking"
# fig1.xlabel = "Timesteps"
# fig1.flg_legend = True
# fig1.figurergba[0] = 0.2
# fig1.figurergba[3] = 0.2
# fig1.gridsize[0] = 5
# fig1.gridsize[1] = 5

# # Add a new figure for orientation tracking
# # Figure 2: End-Effector Orientation Tracking
# viewer.add_line_to_fig(line_name="target_ori_x", fig_idx=2, color=[0.85, 0.3, 0])  # Dark Orange
# viewer.add_line_to_fig(line_name="target_ori_y", fig_idx=2, color=[0.2, 0.6, 0.2])  # Dark Green
# viewer.add_line_to_fig(line_name="target_ori_z", fig_idx=2, color=[0.2, 0.4, 0.8])  # Deep Blue

# viewer.add_line_to_fig(line_name="current_ori_x", fig_idx=2, color=[1, 0.5, 0.2])  # Light Orange
# viewer.add_line_to_fig(line_name="current_ori_y", fig_idx=2, color=[0.4, 0.9, 0.4])  # Light Green
# viewer.add_line_to_fig(line_name="current_ori_z", fig_idx=2, color=[0.3, 0.6, 1])  # Lighter Blue

# fig2 = viewer.figs[2]
# fig2.title = "End-Effector Orientation Tracking"
# fig2.xlabel = "Timesteps"
# fig2.flg_legend = True
# fig2.figurergba[0] = 0.2
# fig2.figurergba[3] = 0.2
# fig2.gridsize[0] = 5
# fig2.gridsize[1] = 5

# Figure 1: Rewards
viewer.add_line_to_fig(line_name="dense_reward", fig_idx=1, color=[0.6, 0, 0])  # Red

fig1 = viewer.figs[1]
fig1.title = "Rewards"
fig1.xlabel = "Timesteps"
fig1.flg_legend = True
fig1.figurergba[0] = 0.2
fig1.figurergba[3] = 0.2
fig1.gridsize[0] = 5
fig1.gridsize[1] = 5


# Figure 2: Joint Velocity
viewer.add_line_to_fig(line_name="sparse_reward", fig_idx=2, color=[0.6, 0, 0])  # Red

fig2 = viewer.figs[2]
fig2.title = "Port Signal"
fig2.xlabel = "Timesteps"
fig2.flg_legend = True
fig2.figurergba[0] = 0.2
fig2.figurergba[3] = 0.2
fig2.gridsize[0] = 5
fig2.gridsize[1] = 5


# Figure 3: Joint Velocity
for joint_idx in ur5e_dof_indices:
    viewer.add_line_to_fig(line_name=f"joint_vel_{joint_idx}", fig_idx=3)

fig3 = viewer.figs[3]
fig3.title = "Joint Velocities"
fig3.xlabel = "Timesteps"
fig3.flg_legend = True
fig3.figurergba[0] = 0.2
fig3.figurergba[3] = 0.2
fig3.gridsize[0] = 5
fig3.gridsize[1] = 5

# Figure 4: Joint Torques
for joint_idx in ur5e_dof_indices:
    viewer.add_line_to_fig(line_name=f"joint_torque_{joint_idx}", fig_idx=4)

fig4 = viewer.figs[4]
fig4.title = "Joint Torques"
fig4.xlabel = "Timesteps"
fig4.flg_legend = True
fig4.figurergba[0] = 0.2
fig4.figurergba[3] = 0.2
fig4.gridsize[0] = 5
fig4.gridsize[1] = 5

# Figure 5: Wrist Force
viewer.add_line_to_fig(line_name="wrist_force_x", fig_idx=5, color=[0.6, 0.1, 0.1])  # Dark Red
viewer.add_line_to_fig(line_name="wrist_force_y", fig_idx=5, color=[0.1, 0.6, 0.1])  # Dark Green
viewer.add_line_to_fig(line_name="wrist_force_z", fig_idx=5, color=[0.1, 0.1, 0.6])  # Dark Blue

fig5 = viewer.figs[5]
fig5.title = "Wrist Force"
fig5.xlabel = "Timesteps"
fig5.flg_legend = True
fig5.figurergba[0] = 0.2
fig5.figurergba[3] = 0.2
fig5.gridsize[0] = 5
fig5.gridsize[1] = 5


# Figure 6: TCP Velocity
viewer.add_line_to_fig(line_name="tcp_vel_x", fig_idx=6, color=[0.8, 0.1, 0.1])  # Red
viewer.add_line_to_fig(line_name="tcp_vel_y", fig_idx=6, color=[0.1, 0.8, 0.1])  # Green
viewer.add_line_to_fig(line_name="tcp_vel_z", fig_idx=6, color=[0.1, 0.1, 0.8])  # Blue

fig6 = viewer.figs[6]
fig6.title = "TCP Velocity"
fig6.xlabel = "Timesteps"
fig6.flg_legend = True
fig6.figurergba[0] = 0.2
fig6.figurergba[3] = 0.2
fig6.gridsize[0] = 5
fig6.gridsize[1] = 5

# Figure 7: Joint Accelerations
for joint_idx in ur5e_dof_indices:
    viewer.add_line_to_fig(line_name=f"joint_acc_{joint_idx}", fig_idx=7)

fig7 = viewer.figs[7]
fig7.title = "Joint Accelerations"
fig7.xlabel = "Timesteps"
fig7.flg_legend = True
fig7.figurergba[0] = 0.2
fig7.figurergba[3] = 0.2
fig7.gridsize[0] = 5
fig7.gridsize[1] = 5

# Main simulation loop
while viewer.is_alive:
    if viewer.reset_requested:
        env.reset()
        action = sample()
        last_sample_time = time.time()
        viewer.reset_requested = False  # Reset the flag

    else:
        step_start = time.time()

        # Update action every 3 seconds
        if time.time() - last_sample_time >= 5.0:
            action = sample()
            last_sample_time = time.time()

        obs, rew, terminated, truncated, info = env.step(action)

        sparse_reward = unwrapped_env.sparse_reward
        dense_reward = unwrapped_env.dense_reward


        # Add marker at mocap position
        mocap_pos = d.mocap_pos[0]
        tcp_pos = d.site_xpos[controller.site_id]
        rotation_matrix = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation_matrix, d.mocap_quat[0])
        # Reshape rotation_matrix to a 3x3 matrix after conversion
        rotation_matrix = rotation_matrix.reshape((3, 3))

        viewer.add_marker(
            pos=mocap_pos,
            mat=rotation_matrix,
            size=[0.001, 0.001, 0.5],
            rgba=[0, 1, 1, 0.3],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
        )

        # Update graph lines for UR5e DOF
        for joint_idx in ur5e_dof_indices:
            viewer.add_data_to_line(line_name=f"qpos_ur5e_joint_{joint_idx}", line_data=d.qpos[joint_idx], fig_idx=0)

        # Update gripper DOF line
        viewer.add_data_to_line(line_name="qpos_gripper", line_data=d.ctrl[gripper_dof_index] / 255, fig_idx=0)

        # # Update target and current position lines
        # viewer.add_data_to_line(line_name="target_x", line_data=mocap_pos[0], fig_idx=1)
        # viewer.add_data_to_line(line_name="target_y", line_data=mocap_pos[1], fig_idx=1)
        # viewer.add_data_to_line(line_name="target_z", line_data=mocap_pos[2], fig_idx=1)
        
        # viewer.add_data_to_line(line_name="current_x", line_data=tcp_pos[0], fig_idx=1)
        # viewer.add_data_to_line(line_name="current_y", line_data=tcp_pos[1], fig_idx=1)
        # viewer.add_data_to_line(line_name="current_z", line_data=tcp_pos[2], fig_idx=1)

        # # Update target and current orientation lines
        # target_quat = d.mocap_quat[0]
        # target = np.zeros(3)
        # mujoco.mju_quat2Vel(target, target_quat, 1)

        # current_rot_mat = d.site_xmat[controller.site_id]
        # current = np.zeros(3)
        # site_quat = np.zeros(4)
        # mujoco.mju_mat2Quat(site_quat, current_rot_mat)
        # mujoco.mju_quat2Vel(current, site_quat, 1)

        # viewer.add_data_to_line(line_name="target_ori_x", line_data=target[0], fig_idx=2)
        # viewer.add_data_to_line(line_name="target_ori_y", line_data=target[1], fig_idx=2)
        # viewer.add_data_to_line(line_name="target_ori_z", line_data=target[2], fig_idx=2)

        # viewer.add_data_to_line(line_name="current_ori_x", line_data=current[0], fig_idx=2)
        # viewer.add_data_to_line(line_name="current_ori_y", line_data=current[1], fig_idx=2)
        # viewer.add_data_to_line(line_name="current_ori_z", line_data=current[2], fig_idx=2)

        # Update rewards plots
        viewer.add_data_to_line(line_name="dense_reward", line_data=dense_reward, fig_idx=1)
        viewer.add_data_to_line(line_name="sparse_reward", line_data=sparse_reward, fig_idx=2)

        # Update Joint Velocity lines
        for joint_idx in ur5e_dof_indices:
            viewer.add_data_to_line(line_name=f"joint_vel_{joint_idx}", line_data=d.qvel[joint_idx], fig_idx=3)

        # Update Joint Torque lines
        for joint_idx in ur5e_dof_indices:
            viewer.add_data_to_line(line_name=f"joint_torque_{joint_idx}", line_data=d.qfrc_actuator[joint_idx], fig_idx=4)

        # Update Wrist Force lines
        _attatchment_id = m.site("attachment_site").id
        bodyid = m.site_bodyid[_attatchment_id]
        rootid = m.body_rootid[bodyid]
        cfrc_int = d.cfrc_int[bodyid]
        total_mass = m.body_subtreemass[bodyid]
        gravity_force = -m.opt.gravity * total_mass
        wrist_force = cfrc_int[3:] - gravity_force

        viewer.add_data_to_line(line_name="wrist_force_x", line_data=wrist_force[0], fig_idx=5)
        viewer.add_data_to_line(line_name="wrist_force_y", line_data=wrist_force[1], fig_idx=5)
        viewer.add_data_to_line(line_name="wrist_force_z", line_data=wrist_force[2], fig_idx=5)

        # Update TCP velocity lines
        tcp_vel = d.sensor("hande/pinch_vel").data
        viewer.add_data_to_line(line_name="tcp_vel_x", line_data=tcp_vel[0], fig_idx=6)
        viewer.add_data_to_line(line_name="tcp_vel_y", line_data=tcp_vel[1], fig_idx=6)
        viewer.add_data_to_line(line_name="tcp_vel_z", line_data=tcp_vel[2], fig_idx=6)

        # Update Joint Acceleration lines
        for joint_idx in ur5e_dof_indices:
            viewer.add_data_to_line(line_name=f"joint_acc_{joint_idx}", line_data=d.qacc[joint_idx], fig_idx=7)

        viewer.render()

        if 'slider_controller' in locals() and slider_controller:
            # Update Tkinter sliders
            slider_controller.root.update_idletasks()
            slider_controller.root.update()

        # Control timestep synchronization
        time_until_next_step = unwrapped_env.control_dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Close viewer after simulation ends
viewer.close()

import os
import time
import copy
import datetime
import pickle as pkl
import numpy as np
import imageio
import gymnasium
import mujoco
import mujoco_sim
from scipy.spatial.transform import Rotation, Slerp

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 1, "Number of successful demos to collect.")
flags.DEFINE_boolean("save_video", False, "Flag to save videos of successful demos.")
flags.DEFINE_boolean("save_model", False, "Flag to save model & data.")

###############################################################################
# Helper Functions
###############################################################################

def compute_waypoints(pos_start, quat_start, pos_end, quat_end, resolution=0.01):
    """
    Returns arrays of position and orientation waypoints
    (linear interpolation for position, slerp for orientation).
    """
    total_distance = np.linalg.norm(pos_end - pos_start)
    N = max(int(total_distance / resolution), 2)  # At least 2 waypoints
    t_vals = np.linspace(0, 1, N)

    # Orientation slerp
    key_times = [0, 1]
    key_rots = Rotation.from_quat([quat_start, quat_end])
    slerp_fn = Slerp(key_times, key_rots)
    interp_rots = slerp_fn(t_vals)
    quat_waypoints = interp_rots.as_quat()

    pos_waypoints = np.linspace(pos_start, pos_end, N)
    return pos_waypoints, quat_waypoints


def apply_action(env, unwrapped_env, trajectory, frames, obs, action):
    """
    Step the environment once, store transition, return next_obs + done info.
    """
    next_obs, rew, done, truncated, info = env.step(action)
    frames.append(unwrapped_env.frames)

    # Store transition
    transition = dict(
        observations=obs,
        actions=copy.deepcopy(action),
        next_observations=copy.deepcopy(next_obs),
        rewards=rew,
        masks=1.0 - done,
        dones=done,
        infos=info,
    )
    trajectory.append(transition)

    return next_obs, done or truncated, info


def move_to_pose(env, unwrapped_env, trajectory, frames, obs,
                 target_pos, target_quat,
                 resolution=0.01,
                 gripper_cmd=0.0,
                 do_wait=True,
                 wait_timeout=3.0,
                 wait_error_thresh=1e-6,
                 sleep_dt=None):
    """
    1) Compute (pos, quat) waypoints from current mocap pose to target.
    2) Step through them, collecting transitions.
    3) Optionally wait for final convergence.
    4) Returns updated (obs, done_flag).
    """
    d = unwrapped_env.data
    curr_pos = d.mocap_pos[0].copy()
    curr_quat = d.mocap_quat[0].copy()

    pos_waypoints, quat_waypoints = compute_waypoints(
        curr_pos, curr_quat,
        target_pos, target_quat,
        resolution=resolution
    )

    # Step through each waypoint
    for i in range(len(pos_waypoints)):
        desired_pos = pos_waypoints[i]
        desired_quat = quat_waypoints[i]

        # Compute delta_pos
        delta_pos = desired_pos - curr_pos

        # Compute orientation error
        quat_conj = np.array([curr_quat[0], -curr_quat[1],
                              -curr_quat[2], -curr_quat[3]])
        orientation_error_quat = np.zeros(4)
        mujoco.mju_mulQuat(orientation_error_quat, desired_quat, quat_conj)
        ori_err = np.zeros(3)
        mujoco.mju_quat2Vel(ori_err, orientation_error_quat, 1.0)

        action = np.concatenate([delta_pos, ori_err, [gripper_cmd]])
        obs, done_flag, info = apply_action(
            env, unwrapped_env, trajectory, frames, obs, action
        )
        if sleep_dt is not None:
            time.sleep(sleep_dt)

        # Update local references
        if done_flag:
            return obs, True  # Episode ended
        curr_pos = desired_pos
        curr_quat = desired_quat

    if do_wait:
        # Wait in place until convergence or timeout
        start_time = time.time()
        while np.linalg.norm(unwrapped_env.controller.error) > wait_error_thresh:
            obs, done_flag, info = apply_action(
                env, unwrapped_env, trajectory, frames, obs, np.zeros(7)
            )
            if done_flag:
                return obs, True
            if (time.time() - start_time) > wait_timeout:
                break

    return obs, False


def retreat_and_offset(env, unwrapped_env,
                       obs, trajectory, frames,
                       retreat_distance=0.02,
                       resolution=0.001,
                       offset_scale=0.01,
                       gripper_cmd=0.0):
    """
    1) Move upward by 'retreat_distance'
    2) Then apply random XY offset (relative to the port bottom, or to current pose)
    Returns (obs, done_flag).
    """
    d = unwrapped_env.data
    curr_pos = d.mocap_pos[0].copy()
    curr_quat = d.mocap_quat[0].copy()

    # 1) Retreat up
    retreat_pos = curr_pos.copy()
    retreat_pos[2] += retreat_distance

    obs, done_flag = move_to_pose(
        env, unwrapped_env, trajectory, frames, obs,
        target_pos=retreat_pos,
        target_quat=curr_quat,
        resolution=resolution,
        gripper_cmd=gripper_cmd,
        do_wait=True,
        wait_timeout=2.0,
        sleep_dt=None
    )
    if done_flag:
        return obs, True

    # 2) Random XY offset
    offset_x = np.random.uniform(-offset_scale, offset_scale)
    offset_y = np.random.uniform(-offset_scale, offset_scale)

    offset_pos = d.mocap_pos[0].copy()
    offset_pos[0] += offset_x
    offset_pos[1] += offset_y

    obs, done_flag = move_to_pose(
        env, unwrapped_env, trajectory, frames, obs,
        target_pos=offset_pos,
        target_quat=curr_quat,
        resolution=resolution,
        gripper_cmd=gripper_cmd,
        do_wait=True,
        wait_timeout=2.0,
        sleep_dt=None
    )
    return obs, done_flag


###############################################################################
# Force-Aware Descent
###############################################################################

def descend_with_force_feedback(env, unwrapped_env,
                                obs, trajectory, frames,
                                port_pos, port_quat,
                                max_retries=10):
    """
    Descend from current pose to port_pos/quat, monitoring z-force.
    If force is too large, retreat and randomize offset, then retry.

    Returns (obs, success_flag).
    """
    d = unwrapped_env.data
    success = False

    for attempt in range(max_retries):
        # Current pose
        curr_pos = d.mocap_pos[0].copy()
        curr_quat = d.mocap_quat[0].copy()

        # Decide a random resolution & threshold for this attempt
        resolution = np.random.uniform(0.0005, 0.0015)
        z_force_threshold = np.random.normal(loc=-20.0, scale=10.0)

        # Compute a single set of waypoints from current to final (keeping XY fixed if desired)
        # Here, we only vary Z from curr_pos[2] -> port_pos[2], or do full:
        pos_waypoints, quat_waypoints = compute_waypoints(
            curr_pos, curr_quat,
            port_pos, port_quat,
            resolution=resolution
        )

        # Step through sub-waypoints
        force_break = False
        for i in range(len(pos_waypoints)):
            desired_pos = pos_waypoints[i]
            desired_quat = quat_waypoints[i]

            # Delta pos & orientation
            delta_pos = desired_pos - curr_pos
            quat_conj = np.array([curr_quat[0], -curr_quat[1],
                                  -curr_quat[2], -curr_quat[3]])
            orientation_error_quat = np.zeros(4)
            mujoco.mju_mulQuat(orientation_error_quat, desired_quat, quat_conj)
            ori_err = np.zeros(3)
            mujoco.mju_quat2Vel(ori_err, orientation_error_quat, 1.0)

            action = np.concatenate([delta_pos, ori_err, [0.0]])
            obs, done_flag, info = apply_action(
                env, unwrapped_env, trajectory, frames, obs, action
            )

            # Update local references
            if done_flag:
                return obs, info.get("succeed", False)
            curr_pos = desired_pos
            curr_quat = desired_quat

            # Check force
            z_force = unwrapped_env.wrist_force[2]
            if z_force < z_force_threshold:
                # Retreat and offset
                obs, done_flag = retreat_and_offset(env, unwrapped_env,
                                                    obs, trajectory, frames,
                                                    retreat_distance=np.random.uniform(0.005, 0.02),
                                                    resolution=resolution,
                                                    offset_scale=0.01)
                if done_flag:
                    return obs, True
                force_break = True
                break

        if not force_break:
            # We completed all waypoints without exceeding force threshold
            print(f"Descent succeeded on attempt {attempt+1}")
            success = True
            break

    # If we reach here, either success or gave up after max_retries
    return obs, success


###############################################################################
# Main Demo Routine
###############################################################################

def run_demo(env, unwrapped_env, obs, trajectory, frames):
    """
    Example multi-stage demo:
      1. Move to "hover" pose above the port
      2. Force-aware descent (with multiple retries)
      3. Return (trajectory, success_flag)
    """
    d = unwrapped_env.data

    # 1) Move to a random hover pose above the port
    port_bottom_pos = d.sensor("port_bottom_pos").data.copy()
    port_bottom_quat = d.sensor("port_bottom_quat").data.copy()

    hover_pos = port_bottom_pos.copy()
    hover_pos[0] += np.random.uniform(-0.01, 0.01)
    hover_pos[1] += np.random.uniform(-0.01, 0.01)
    hover_pos[2] += np.random.uniform(0.05, 0.10)  # Hover height

    obs, done_flag = move_to_pose(
        env, unwrapped_env, trajectory, frames, obs,
        target_pos=hover_pos,
        target_quat=port_bottom_quat,
        resolution=0.001,
        gripper_cmd=0.0,
        do_wait=True,
        wait_timeout=10.0,
        wait_error_thresh=1e-6,
        sleep_dt= unwrapped_env.control_dt
    )
    if done_flag:
        return trajectory, True

    # 2) Force-aware descent
    obs, descent_success = descend_with_force_feedback(
        env, unwrapped_env,
        obs, trajectory, frames,
        port_pos=port_bottom_pos,
        port_quat=port_bottom_quat,
        max_retries=20
    )
    if descent_success:
        # Optionally wait for final convergence
        obs, done_flag = move_to_pose(
            env, unwrapped_env, trajectory, frames, obs,
            target_pos=port_bottom_pos,
            target_quat=port_bottom_quat,
            resolution=0.001,
            gripper_cmd=0.0,
            do_wait=True,
            wait_timeout=10.0,
            wait_error_thresh=1e-6,
            sleep_dt=unwrapped_env.control_dt
        )
        if done_flag:
            descent_success = True

    return trajectory, descent_success


###############################################################################
# If running as main
###############################################################################
def main(_):
    # Example usage
    env = gymnasium.make("ur5ePegInHoleFixedGymEnv_state-v0", render_mode="human")
    unwrapped_env = env.unwrapped

    successes_collected = 0
    n_needed = FLAGS.successes_needed
    all_demos = []

    while successes_collected < n_needed:
        obs, info = env.reset()
        trajectory = []
        frames = []

        trajectory, success = run_demo(env, unwrapped_env, obs, trajectory, frames)

        if success:
            successes_collected += 1
            all_demos.append(trajectory)
            print(f"Demo success! Total successes: {successes_collected}/{n_needed}")

        # Optionally save videos or model here
        if FLAGS.save_video and success:
            tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"{FLAGS.exp_name}_demo_{tstamp}.mp4"
            # Flatten frames list (they might be lists-of-lists)
            flat_frames = []
            for f in frames:
                if isinstance(f, list):
                    flat_frames.extend(f)
                else:
                    flat_frames.append(f)
            imageio.mimsave(video_path, flat_frames, fps=20)
            print(f"Saved video to: {video_path}")

    if FLAGS.save_model:
        # Save demonstrations, model, etc.
        tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = f"{FLAGS.exp_name}_demos_{tstamp}.pkl"
        with open(data_path, "wb") as f:
            pkl.dump(all_demos, f)
        print(f"Saved demos to: {data_path}")


if __name__ == "__main__":
    app.run(main)

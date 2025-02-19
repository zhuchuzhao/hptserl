import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import time
import imageio
import gymnasium
import mujoco
import mujoco_sim
from scipy.spatial.transform import Rotation, Slerp

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful demos to collect.")
flags.DEFINE_boolean("save_video", True, "Flag to save videos of successful demos.")
flags.DEFINE_boolean("save_model", True, "Flag to save model & data.")
flags.DEFINE_float("noise_std", 0.0005, "Standard deviation for noise to add to offsets.")

################################################################################
# Helper Functions
################################################################################

def compute_waypoints(pos_start, quat_start, pos_end, quat_end, resolution=0.0005):
    """
    Returns arrays of position and orientation waypoints
    (linear interpolation for position, slerp for orientation).
    """
    total_distance = np.linalg.norm(pos_end - pos_start)
    N = max(int(total_distance / resolution), 2)  # Ensure at least 2 waypoints
    t_vals = np.linspace(0, 1, N)
    key_times = [0, 1]

    key_rots = Rotation.from_quat([quat_start, quat_end])
    slerp_fn = Slerp(key_times, key_rots)
    interp_rots = slerp_fn(t_vals)
    quat_waypoints = interp_rots.as_quat()

    pos_waypoints = np.linspace(pos_start, pos_end, N)
    return pos_waypoints, quat_waypoints


def step_through_waypoints(env, unwrapped_env,
                           pos_waypoints, quat_waypoints,
                           trajectory, frames, obs,
                           gripper_cmd=0.0):
    """
    Step the environment through the given (pos, quat) waypoints.
    Updates 'trajectory' and 'frames', returns the latest obs + success flag.
    """
    prev_pos = pos_waypoints[0]
    prev_quat = quat_waypoints[0]

    for i in range(len(pos_waypoints)):
        desired_pos = pos_waypoints[i]
        desired_quat = quat_waypoints[i]

        # Delta position
        delta_pos = desired_pos - prev_pos

        # Delta orientation
        quat_conj = np.array([prev_quat[0], -prev_quat[1], -prev_quat[2], -prev_quat[3]])
        orientation_error_quat = np.zeros(4)
        mujoco.mju_mulQuat(orientation_error_quat, desired_quat, quat_conj)
        ori_err = np.zeros(3)
        mujoco.mju_quat2Vel(ori_err, orientation_error_quat, 1.0)

        # action = np.concatenate([delta_pos, ori_err, [gripper_cmd]])
        action = delta_pos

        next_obs, rew, done, truncated, info = env.step(action)
        frames.append(unwrapped_env.frames)

        # Store transition
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=action,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)

        obs = next_obs
        prev_pos = desired_pos
        prev_quat = desired_quat


        if done or truncated:
            # Episode ended prematurely
            return obs, info.get("succeed", False)

    return obs, info.get("succeed", False)


def wait_for_convergence(env, unwrapped_env, trajectory, frames, obs,
                         timeout=1.0, error_threshold=1e-4):
    """
    Wait in place until controller error is minimal or until timeout/done.
    """
    start_time = time.time()
    info = {}
    while np.linalg.norm(unwrapped_env.controller.error) > error_threshold:
        next_obs, rew, done, truncated, info = env.step(np.zeros(3))
        frames.append(unwrapped_env.frames)

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=np.zeros(3),
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info
            )
        )
        trajectory.append(transition)
        obs = next_obs

        if done or truncated:
            return obs, info.get("succeed", False)

        if (time.time() - start_time) > timeout:
            break

    return obs, info.get("succeed", False)

################################################################################
# Force-Aware Descent with “Retreat & Re-Offset” Logic
################################################################################

def descend_with_force_feedback(env, unwrapped_env,
                                obs, trajectory, frames,
                                port_pos, port_quat,
                                max_retries=50,
                                force_check_duration=0.5,  # Duration in seconds
                                force_check_interval=0.01  # Interval in seconds
                                ):
    """
    Attempt to descend from current pose to the port bottom pose, 
    monitoring force along the way. If excessive force is detected
    continuously for a specified duration, retreat and randomly 
    re-offset the approach pose, then try again.

    - port_pos, port_quat: Target pose (bottom of the port).
    - max_retries: How many times we allow re-offset attempts.
    - force_check_duration: Time duration the force must remain below the threshold.
    - force_check_interval: Time between force checks.
    
    Returns: obs, success_flag
    """
    d = unwrapped_env.data
    success = False
    accumulated_time = 0.0  # Initialize the accumulated time

    for attempt in range(max_retries):
        # ---------------------------------------------------------------------
        # 1) Compute small, incremental waypoints from current to target
        #    We'll take short steps so we can read force feedback at each step.
        # ---------------------------------------------------------------------
        current_pos = d.mocap_pos[0].copy()
        current_quat = d.mocap_quat[0].copy()

        end_pos = current_pos.copy()
        end_pos[2] = port_pos[2]  # Keep z-coordinate fixed

        z_force_threshold = np.random.uniform(-5, -50)
        resolution = 0.0005

        # Create a fine-grained set of waypoints for the descent
        pos_waypoints, quat_waypoints = compute_waypoints(
            pos_start=current_pos,
            quat_start=current_quat,
            pos_end=end_pos,
            quat_end=port_quat,
            resolution=resolution
        )

        # We'll step manually through each sub-waypoint
        for i in range(len(pos_waypoints)):
            desired_pos = pos_waypoints[i]
            desired_quat = quat_waypoints[i]

            # ---------------------------------------------
            # Compute action (delta_pos, delta_ori, gripper)
            # ---------------------------------------------
            delta_pos = desired_pos - current_pos
            quat_conj = np.array([current_quat[0], -current_quat[1],
                                  -current_quat[2], -current_quat[3]])
            orientation_error_quat = np.zeros(4)
            mujoco.mju_mulQuat(orientation_error_quat, desired_quat, quat_conj)
            ori_err = np.zeros(3)
            mujoco.mju_quat2Vel(ori_err, orientation_error_quat, 1.0)

            # action = np.concatenate([delta_pos, ori_err, [0.0]])  # 0.0 for gripper
            action = delta_pos

            next_obs, rew, done, truncated, info = env.step(action)
            frames.append(unwrapped_env.frames)

            # Store transition
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=action,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)

            obs = next_obs
            current_pos = desired_pos
            current_quat = desired_quat


            if done or truncated:
                return obs, info.get("succeed", False)

            z_force = unwrapped_env.wrist_force[2]

            # Check if z_force is below the threshold
            if z_force < z_force_threshold:
                accumulated_time += force_check_interval
                if accumulated_time >= force_check_duration:
                    # --- TAKE A SMALL REST ---
                    for _ in range(30):
                        # Zero action → no movement
                        r_obs, r_rew, r_done, r_truncated, r_info = env.step(np.zeros(3))
                        
                        # It's good practice to log transitions during rest
                        rest_transition = copy.deepcopy(
                            dict(
                                observations=obs,
                                actions=np.zeros(3),
                                next_observations=r_obs,
                                rewards=r_rew,
                                masks=1.0 - r_done,
                                dones=r_done,
                                infos=r_info,
                            )
                        )
                        trajectory.append(rest_transition)
                        obs = r_obs
                        frames.append(unwrapped_env.frames)

                        if r_done or r_truncated:
                            return obs, r_info.get("succeed", False)
                    # Retreat upward by 'retreat_distance'
                    retreat_pos = d.mocap_pos[0].copy()
                    conn_pos = d.site_xpos[unwrapped_env._pinch_site_id]
                    error = conn_pos[2] - retreat_pos[2]

                    retreat_distance = np.random.uniform(0.005, 0.007)  # Example range: 5 cm to 10 cm
                    retreat_pos[2] += retreat_distance
                    retreat_pos[2] += error
                    retreat_pos[2] += np.random.normal(0, FLAGS.noise_std)  # add Gaussian noise

                    # Step through retreat
                    ret_waypoints, ret_quat = compute_waypoints(
                        pos_start=d.mocap_pos[0].copy(),
                        quat_start=d.mocap_quat[0].copy(),
                        pos_end=retreat_pos,
                        quat_end=current_quat,
                        resolution=resolution
                    )
                    obs, ended = step_through_waypoints(
                        env, unwrapped_env,
                        ret_waypoints, ret_quat,
                        trajectory, frames, obs,
                        gripper_cmd=0.0
                    )
                    if ended:
                        return obs, True  # environment ended
                    # Wait for convergence after retreat
                    obs, ended = wait_for_convergence(env, unwrapped_env,
                                                     trajectory, frames, obs,
                                                     timeout=1.0)
                    if ended:
                        return obs, True  # environment ended

                    # Randomly re-offset the final target again
                    _retreat_pos = d.mocap_pos[0].copy()
                    original_offset = port_pos[0] - _retreat_pos[0]  # Assuming port_pos has at least 2 elements
                    offset_x = np.random.uniform(0, abs(original_offset)) * np.sign(original_offset)
                    noise_x = np.random.normal(0, FLAGS.noise_std)
                    offset_x += abs(noise_x) * np.sign(original_offset)
                    _retreat_pos[0] += offset_x

                    # Step through re-offset
                    ret_waypoints, ret_quat = compute_waypoints(
                        pos_start=d.mocap_pos[0].copy(),
                        quat_start=d.mocap_quat[0].copy(),
                        pos_end=_retreat_pos,
                        quat_end=current_quat,
                        resolution=resolution
                    )
                    obs, ended = step_through_waypoints(
                        env, unwrapped_env,
                        ret_waypoints, ret_quat,
                        trajectory, frames, obs,
                        gripper_cmd=0.0
                    )
                    if ended:
                        return obs, True  # environment ended
                    # Wait for convergence after re-offset
                    obs, ended = wait_for_convergence(env, unwrapped_env,
                                                     trajectory, frames, obs,
                                                     timeout=1.0)
                    if ended:
                        return obs, True  # environment ended
                    # Randomly re-offset the final target again
                    _retreat_pos = d.mocap_pos[0].copy()
                    original_offset = port_pos[1] - _retreat_pos[1]  # Assuming port_pos has at least 2 elements
                    offset_y = np.random.uniform(0, abs(original_offset)) * np.sign(original_offset)
                    noise_y = np.random.normal(0, FLAGS.noise_std)
                    offset_y += abs(noise_y) * np.sign(original_offset)
                    _retreat_pos[1] += offset_y

                    # Step through re-offset
                    ret_waypoints, ret_quat = compute_waypoints(
                        pos_start=d.mocap_pos[0].copy(),
                        quat_start=d.mocap_quat[0].copy(),
                        pos_end=_retreat_pos,
                        quat_end=current_quat,
                        resolution=resolution
                    )
                    obs, ended = step_through_waypoints(
                        env, unwrapped_env,
                        ret_waypoints, ret_quat,
                        trajectory, frames, obs,
                        gripper_cmd=0.0
                    )
                    if ended:
                        return obs, True  # environment ended
                    # Wait for convergence after re-offset
                    obs, ended = wait_for_convergence(env, unwrapped_env,
                                                     trajectory, frames, obs,
                                                     timeout=1.0)
                    if ended:
                        return obs, True  # environment ended

                    # Reset the accumulated time after retreat
                    accumulated_time = 0.0

                    # Break out of this sub-waypoint loop and retry descent
                    break
            else:
                # Reset the accumulated time if force is above the threshold
                accumulated_time = 0.0
        else:
            # If we didn't 'break' from the loop, it means we
            # got through all waypoints without force issues.
            print(f"Descent succeeded on attempt {attempt+1}")
            success = True
            break

    return obs, success


################################################################################
# Main Demo Routine
################################################################################

def run_demo(env, unwrapped_env, obs, trajectory, frames):
    """
    Example multi-stage demo:
      1. Randomly offset above the port (a "hover" pose)
      2. Descend with force feedback (possibly multiple retries)
      3. End
    """
    d = unwrapped_env.data

    # ------------------------------------------------------------------
    # 1) Move to a random "hover pose" above the port
    # ------------------------------------------------------------------
    port_top_pos = d.sensor("port_top_pos").data.copy()
    port_bottom_pos = d.sensor("port_bottom_pos").data.copy()

    port_bottom_quat = d.sensor("port_bottom_quat").data.copy()

    hover_pos = d.mocap_pos[0].copy()
    # Compute directional offsets based on the current retreat position
    original_offset_z = hover_pos[2] - port_top_pos[2] - 0.05
    # Restrict offset direction to follow the original offset direction
    offset_z = np.random.uniform(0, abs(original_offset_z)) * np.sign(original_offset_z)
    offset_z += np.random.normal(0, FLAGS.noise_std)  # add Gaussian noise
    hover_pos[2] -= offset_z

    # Waypoints to hover pose
    stage1_pos_waypoints, stage1_quat_waypoints = compute_waypoints(
        pos_start=d.mocap_pos[0].copy(),
        quat_start=d.mocap_quat[0].copy(),
        pos_end=hover_pos,
        quat_end=port_bottom_quat,  # Align with port orientation
        resolution=0.001
    )

    # Execute Stage 1
    obs, success = step_through_waypoints(
        env, unwrapped_env,
        stage1_pos_waypoints, stage1_quat_waypoints,
        trajectory, frames, obs
    )
    if success:
        return trajectory, True  # environment ended

    # Wait for final convergence after Stage 1
    obs, success = wait_for_convergence(
        env, unwrapped_env,
        trajectory, frames, obs,
        timeout=1.0
    )
    if success:
        return trajectory, True
    
    hover_pos = d.mocap_pos[0].copy()
    original_offset_x = hover_pos[0] - port_bottom_pos[0]
    upper_offset = abs(original_offset_x) * np.sign(original_offset_x)
    offset_x = np.random.uniform(upper_offset/2, upper_offset)
    offset_x += np.random.normal(0, FLAGS.noise_std)  # add Gaussian noise
    hover_pos[0] -= offset_x

    # Waypoints to hover pose
    stage1_pos_waypoints, stage1_quat_waypoints = compute_waypoints(
        pos_start=d.mocap_pos[0].copy(),
        quat_start=d.mocap_quat[0].copy(),
        pos_end=hover_pos,
        quat_end=port_bottom_quat,  # Align with port orientation
        resolution=0.001
    )

    # Execute Stage 1
    obs, success = step_through_waypoints(
        env, unwrapped_env,
        stage1_pos_waypoints, stage1_quat_waypoints,
        trajectory, frames, obs
    )
    if success:
        return trajectory, True  # environment ended

    # Wait for final convergence after Stage 1
    obs, success = wait_for_convergence(
        env, unwrapped_env,
        trajectory, frames, obs,
        timeout=1.0
    )
    if success:
        return trajectory, True

    hover_pos = d.mocap_pos[0].copy()
    original_offset_y = hover_pos[1] - port_bottom_pos[1]
    upper_offset = abs(original_offset_y) * np.sign(original_offset_y)
    offset_y = np.random.uniform(upper_offset/2, upper_offset)
    offset_y += np.random.normal(0, FLAGS.noise_std)  # add Gaussian noise

    hover_pos[1] -= offset_y

    # Waypoints to hover pose
    stage1_pos_waypoints, stage1_quat_waypoints = compute_waypoints(
        pos_start=d.mocap_pos[0].copy(),
        quat_start=d.mocap_quat[0].copy(),
        pos_end=hover_pos,
        quat_end=port_bottom_quat,  # Align with port orientation
        resolution=0.001
    )

    # Execute Stage 1
    obs, success = step_through_waypoints(
        env, unwrapped_env,
        stage1_pos_waypoints, stage1_quat_waypoints,
        trajectory, frames, obs
    )
    if success:
        return trajectory, True  # environment ended

    # Wait for final convergence after Stage 1
    obs, success = wait_for_convergence(
        env, unwrapped_env,
        trajectory, frames, obs,
        timeout=1.0
    )
    if success:
        return trajectory, True

    # ------------------------------------------------------------------
    # 2) Force-aware descent, possibly with multiple retries
    # ------------------------------------------------------------------
    obs, success = descend_with_force_feedback(
        env, unwrapped_env,
        obs, trajectory, frames,
        port_pos=port_bottom_pos,
        port_quat=port_bottom_quat,
        max_retries=20,
        force_check_duration=0.1,  # e.g., 0.5 seconds
        force_check_interval=unwrapped_env.control_dt  # Ensure this matches your control loop
    )
    if success:
        return trajectory, True
    
    return trajectory, False


def main(_):
    env = gymnasium.make("ur5ePegInHoleFixedGymEnv_state-v0", render_mode="human")
    # env = gymnasium.make("ur5ePegInHoleFixedGymEnv_state-v0")

    unwrapped_env = env.unwrapped
    action_spec = env.action_space
    print(f"Action space: {action_spec}")

    observation_spec = env.observation_space
    print(f"Observation space: {observation_spec}")
    obs, info = env.reset()
    print("Reset done")

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    frames = []
    trajectory = []
    returns = 0

    while success_count < success_needed:
        # Execute the multi-stage, force-aware demo
        trajectory, success = run_demo(env, unwrapped_env, obs, trajectory, frames)

        # Compute episode return
        returns = sum(t["rewards"] for t in trajectory)

        # If successful
        if success:
            pbar.set_description(f"Episode Return: {returns:.2f} [SUCCESS]")
            success_count += 1
            pbar.update(1)

            # Save entire trajectory
            transitions += copy.deepcopy(trajectory)

            if FLAGS.save_video:
                video_name = f"./demo_data/success_demo_{success_count}.mp4"
                imageio.mimsave(video_name, frames, fps=20)
                print(f"Saved video to {video_name}")
        else:
            pbar.set_description(f"Episode Return: {returns:.2f} [FAILED]")

        # Reset for next iteration
        trajectory.clear()
        frames.clear()
        returns = 0
        obs, info = env.reset()

    # Once we have all successful demos, optionally save them
    if FLAGS.save_model:
        if not os.path.exists("./demo_data"):
            os.makedirs("./demo_data")
        uuid_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid_str}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(transitions, f)
        print(f"Saved {success_needed} successful demos to {file_name}")

    env.close()


if __name__ == "__main__":
    app.run(main)

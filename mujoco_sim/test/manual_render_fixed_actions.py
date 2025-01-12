import numpy as np
import time
import gymnasium
import mujoco
import mujoco_sim

from scipy.spatial.transform import Rotation, Slerp

def compute_waypoints(pos_start, quat_start, pos_end, quat_end, N):
    """
    Returns position and orientation waypoints (with linear interpolation of position
    and slerp for orientation).
    """
    # Times for interpolation
    t_vals = np.linspace(0, 1, N)
    key_times = [0, 1]

    # Orientation via slerp
    key_rots = Rotation.from_quat([quat_start, quat_end])
    slerp_fn = Slerp(key_times, key_rots)
    interp_rots = slerp_fn(t_vals)
    quat_waypoints = interp_rots.as_quat()

    # Position (linear interpolation)
    pos_waypoints = np.linspace(pos_start, pos_end, N)

    return pos_waypoints, quat_waypoints


def has_reached_pose(
    current_pos, current_quat, target_pos, target_quat,
    pos_threshold=0.001, angle_threshold_deg=1.0
):
    """
    Check if the current pose is within a certain position/orientation threshold
    from the target pose.

    :param current_pos: np.array(3,) - current TCP position
    :param current_quat: np.array(4,) - current TCP quaternion (x, y, z, w)
    :param target_pos: np.array(3,) - target TCP position
    :param target_quat: np.array(4,) - target TCP quaternion (x, y, z, w)
    :param pos_threshold: float, linear distance threshold (meters)
    :param angle_threshold_deg: float, angular threshold (degrees)
    :return: True if within thresholds, False otherwise
    """
    # Check position error
    pos_error = np.linalg.norm(current_pos - target_pos)
    if pos_error > pos_threshold:
        return False

    # Check orientation error by quaternion distance
    rot_current = Rotation.from_quat(current_quat)
    rot_target = Rotation.from_quat(target_quat)
    # Relative rotation
    rot_diff = rot_target * rot_current.inv()
    angle_deg = np.degrees(rot_diff.magnitude())
    if angle_deg > angle_threshold_deg:
        return False

    return True


if __name__ == "__main__":

    # --- Number of waypoints per stage ---
    N_stage1 = 10
    N_stage2 = 10

    # --- Create and reset the environment ---
    env = gymnasium.make("ur5ePegInHoleFixedGymEnv_state-v0", render_mode="human")
    unwrapped_env = env.unwrapped
    m = unwrapped_env.model
    d = unwrapped_env.data
    env.reset()

    # Initial steps with zero action
    for _ in range(100):
        obs, rew, done, truncated, info = env.step(np.zeros(7))

    # --- Retrieve the initial pose ---
    initial_pos = d.sensor("connector_bottom_pos").data.copy()
    initial_quat = d.sensor("connector_head_quat").data.copy()

    # --- Retrieve the target pose ---
    port_bottom_pos = d.sensor("port_bottom_pos").data.copy()
    port_bottom_quat = d.sensor("port_bottom_quat").data.copy()

    # --------------------------------------------------
    # 1) Define an intermediate pose "above" the port
    # --------------------------------------------------
    random_offset_z = np.random.uniform(0.02, 0.05)
    above_port_pos = port_bottom_pos.copy()
    above_port_pos[2] += random_offset_z

    # --------------------------------------------------
    # 2) STAGE 1: from initial pose to above_port_pos
    # --------------------------------------------------
    stage1_pos_waypoints, stage1_quat_waypoints = compute_waypoints(
        pos_start=initial_pos,
        quat_start=initial_quat,
        pos_end=above_port_pos,
        quat_end=port_bottom_quat,
        N=N_stage1
    )

    # --- Execute Stage 1 ---
    prev_pos = initial_pos
    prev_quat = initial_quat
    gripper_cmd = 0.0

    print("===== STAGE 1: Moving above port =====")
    for i in range(N_stage1):
        current_time = time.time()

        desired_pos = stage1_pos_waypoints[i]
        desired_quat = stage1_quat_waypoints[i]

        # Delta position
        delta_pos = desired_pos - prev_pos

        # Delta orientation
        quat_conj = np.array([prev_quat[0], -prev_quat[1], -prev_quat[2], -prev_quat[3]])
        orientation_error_quat = np.zeros(4)
        mujoco.mju_mulQuat(orientation_error_quat, desired_quat, quat_conj)
        ori_err = np.zeros(3)
        mujoco.mju_quat2Vel(ori_err, orientation_error_quat, 1.0)

        action = np.concatenate([delta_pos, ori_err, [gripper_cmd]])
        print(f"[STAGE 1] Waypoint {i+1}/{N_stage1}, Action = {action}")

        obs, rew, done, truncated, info = env.step(action)

        # Update for next iteration
        prev_pos = desired_pos
        prev_quat = desired_quat

        # Sleep until next control step
        time_until_next_step = unwrapped_env.control_dt - (time.time() - current_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        if done or truncated:
            print("Environment ended prematurely during Stage 1.")
            obs, info = env.reset()
            break

    # ------------------------------------------------------------------
    # Wait until the TCP has truly reached the Stage 1 final pose
    # (within some threshold), or until we exceed a timeout.
    # ------------------------------------------------------------------
    stage1_target_pos = above_port_pos
    stage1_target_quat = port_bottom_quat

    print("Waiting until Stage 1 final pose is reached...")
    start_wait_time = time.time()
    WAIT_TIMEOUT = 5.0  # seconds
    reached_stage_1 = False

    while True:
        current_pos = d.sensor("connector_bottom_pos").data.copy()
        current_quat = d.sensor("connector_head_quat").data.copy()

        if has_reached_pose(current_pos, current_quat, stage1_target_pos, stage1_target_quat):
            reached_stage_1 = True
            break

        # Step the environment in place with zero action while waiting
        obs, rew, done, truncated, info = env.step(np.zeros(7))
        if done or truncated:
            print("Environment ended prematurely during the waiting period.")
            env.reset()
            break

        if (time.time() - start_wait_time) > WAIT_TIMEOUT:
            # Compute position and orientation error for logging
            pos_error = np.linalg.norm(current_pos - stage1_target_pos)

            rot_current = Rotation.from_quat(current_quat)
            rot_target = Rotation.from_quat(stage1_target_quat)
            rot_diff = rot_target * rot_current.inv()
            angle_deg = np.degrees(rot_diff.magnitude())

            print("Timeout reached while waiting for Stage 1 pose.")
            print(f"Position error: {pos_error:.4f} m")
            print(f"Orientation error: {angle_deg:.2f} deg")
            break

    if reached_stage_1:
        print("Stage 1 complete: TCP is above the port. Proceeding to Stage 2.")
        for _ in range(50):
            obs, rew, done, truncated, info = env.step(np.zeros(7))
    else:
        print("Warning: Stage 1 final pose NOT reached within threshold/time. Proceeding anyway...")
        for _ in range(50):
            obs, rew, done, truncated, info = env.step(np.zeros(7))

    # --- Execute Stage 2 only after Stage 1 is confirmed (or forced) ---
    print("===== STAGE 2: Descending into port =====")

        # --------------------------------------------------
    # 3) STAGE 2: from above_port_pos down to port_bottom_pos
    # --------------------------------------------------
        # --- Retrieve the initial pose ---
    initial_pos = d.sensor("connector_bottom_pos").data.copy()
    initial_quat = d.sensor("connector_bottom_quat").data.copy()

    # --- Retrieve the target pose ---
    port_bottom_pos = d.sensor("port_bottom_pos").data.copy()
    port_bottom_quat = d.sensor("port_bottom_quat").data.copy()

    stage2_pos_waypoints, stage2_quat_waypoints = compute_waypoints(
        pos_start=initial_pos,
        quat_start=port_bottom_quat,  # orientation from stage1 end
        pos_end=port_bottom_pos,
        quat_end=port_bottom_quat,    # final orientation
        N=N_stage2
    )

    # Update our "previous pose" in case there's been motion during the wait
    prev_pos = d.sensor("connector_bottom_pos").data.copy()
    prev_quat = d.sensor("connector_head_quat").data.copy()

    for i in range(N_stage2):
        current_time = time.time()

        desired_pos = stage2_pos_waypoints[i]
        desired_quat = stage2_quat_waypoints[i]

        # Delta position
        delta_pos = desired_pos - prev_pos

        # Delta orientation
        quat_conj = np.array([prev_quat[0], -prev_quat[1], -prev_quat[2], -prev_quat[3]])
        orientation_error_quat = np.zeros(4)
        mujoco.mju_mulQuat(orientation_error_quat, desired_quat, quat_conj)
        ori_err = np.zeros(3)
        mujoco.mju_quat2Vel(ori_err, orientation_error_quat, 1.0)

        action = np.concatenate([delta_pos, ori_err, [gripper_cmd]])
        print(f"[STAGE 2] Waypoint {i+1}/{N_stage2}, Action = {action}")

        obs, rew, done, truncated, info = env.step(action)

        prev_pos = desired_pos
        prev_quat = desired_quat

        # Sleep for the next control step
        time_until_next_step = unwrapped_env.control_dt - (time.time() - current_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        if done or truncated:
            print("Environment ended prematurely during Stage 2.")
            obs, info = env.reset()
            break

    print("Reached final pose. Robot will remain at the last pose. Press CTRL+C to exit.")

    # Keep the environment alive with zero actions
    while True:
        obs, rew, done, truncated, info = env.step(np.zeros(7))
        if done or truncated:
            obs, info = env.reset()

    env.close()

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
flags.DEFINE_integer("successes_needed", 1, "Number of successful demos to collect.")
flags.DEFINE_boolean("save_video", True, "Flag to save videos of successful demos.")


def compute_waypoints(pos_start, quat_start, pos_end, quat_end, N):
    """
    Returns position and orientation waypoints (with linear interpolation of position
    and slerp for orientation).
    """
    t_vals = np.linspace(0, 1, N)
    key_times = [0, 1]

    key_rots = Rotation.from_quat([quat_start, quat_end])
    slerp_fn = Slerp(key_times, key_rots)
    interp_rots = slerp_fn(t_vals)
    quat_waypoints = interp_rots.as_quat()

    pos_waypoints = np.linspace(pos_start, pos_end, N)
    return pos_waypoints, quat_waypoints


def has_reached_pose(
    current_pos, current_quat, target_pos, target_quat,
    pos_threshold=0.001, angle_threshold_deg=1.0
):
    """
    Check if the current pose is within a certain threshold (position + orientation).
    """
    pos_error = np.linalg.norm(current_pos - target_pos)
    if pos_error > pos_threshold:
        return False

    rot_current = Rotation.from_quat(current_quat)
    rot_target = Rotation.from_quat(target_quat)
    rot_diff = rot_target * rot_current.inv()
    angle_deg = np.degrees(rot_diff.magnitude())
    return angle_deg <= angle_threshold_deg


def run_two_stage_demo(env, unwrapped_env, frames=None):
    """
    Example two-stage motion approach:
     1) Move from initial pose to an 'above port' pose
     2) Then move from that 'above port' pose down into the port

    Returns:
      - transitions: list of (obs, act, next_obs, reward, done, info) dicts
      - success: whether we ended with info["succeed"] == True
    """

    d = unwrapped_env.data

    # Observations and transitions
    obs_list = []
    act_list = []
    next_obs_list = []
    rew_list = []
    done_list = []
    info_list = []

    # --------------------------------------------------
    #  Retrieve the initial pose
    # --------------------------------------------------
    initial_pos = d.sensor("connector_bottom_pos").data.copy()
    initial_quat = d.sensor("connector_head_quat").data.copy()

    # --------------------------------------------------
    #  Retrieve the target pose
    # --------------------------------------------------
    port_bottom_pos = d.sensor("port_bottom_pos").data.copy()
    port_bottom_quat = d.sensor("port_bottom_quat").data.copy()

    # --------------------------------------------------
    # 1) Define an intermediate pose "above" the port
    # --------------------------------------------------
    random_offset_z = np.random.uniform(0.02, 0.05)
    above_port_pos = port_bottom_pos.copy()
    above_port_pos[2] += random_offset_z

    # --------------------------------------------------
    # 2) Compute Stage 1 waypoints
    # --------------------------------------------------
    N_stage1 = 10
    stage1_pos_waypoints, stage1_quat_waypoints = compute_waypoints(
        pos_start=initial_pos,
        quat_start=initial_quat,
        pos_end=above_port_pos,
        quat_end=port_bottom_quat,
        N=N_stage1
    )

    # Execute Stage 1
    prev_pos = initial_pos
    prev_quat = initial_quat
    gripper_cmd = 0.0

    for i in range(N_stage1):
        desired_pos = stage1_pos_waypoints[i]
        desired_quat = stage1_quat_waypoints[i]

        # Delta position
        delta_pos = desired_pos - prev_pos

        # Delta orientation (using Mujoco’s quaternion utility)
        quat_conj = np.array([prev_quat[0], -prev_quat[1], -prev_quat[2], -prev_quat[3]])
        orientation_error_quat = np.zeros(4)
        mujoco.mju_mulQuat(orientation_error_quat, desired_quat, quat_conj)
        ori_err = np.zeros(3)
        mujoco.mju_quat2Vel(ori_err, orientation_error_quat, 1.0)

        action = np.concatenate([delta_pos, ori_err, [gripper_cmd]])

        # Step environment
        next_obs, rew, done, truncated, info = env.step(action)
        obs = next_obs  # or keep track of the last obs

        if frames is not None:
            # If your obs contains images directly:
            if "front" in next_obs and "wrist" in next_obs:
                frames.append(
                    np.concatenate((next_obs["front"], next_obs["wrist"]), axis=0)
                )
            else:
                # Otherwise, render from the environment
                frame1, frame2 = env.render()
                frames.append(np.concatenate((frame1, frame2), axis=0))

        # Store transition
        transition = dict(
            observations=copy.deepcopy(obs),
            actions=copy.deepcopy(action),
            next_observations=copy.deepcopy(next_obs),
            rewards=rew,
            dones=done,
            infos=info
        )
        obs_list.append(transition["observations"])
        act_list.append(transition["actions"])
        next_obs_list.append(transition["next_observations"])
        rew_list.append(transition["rewards"])
        done_list.append(transition["dones"])
        info_list.append(transition["infos"])

        # Update for next iteration
        prev_pos = desired_pos
        prev_quat = desired_quat

        # Sleep until next control step
        time.sleep(unwrapped_env.control_dt)

        if done or truncated:
            # Episode ended prematurely
            return (obs_list, act_list, next_obs_list, rew_list, done_list, info_list), info.get("succeed", False)

    # Wait until final pose is reached or until we exceed a timeout
    start_wait_time = time.time()
    WAIT_TIMEOUT = 5.0

    stage1_target_pos = above_port_pos
    stage1_target_quat = port_bottom_quat

    while True:
        current_pos = d.sensor("connector_bottom_pos").data.copy()
        current_quat = d.sensor("connector_head_quat").data.copy()

        if has_reached_pose(current_pos, current_quat, stage1_target_pos, stage1_target_quat):
            break

        # Step in place with zero action while waiting
        obs = next_obs
        next_obs, rew, done, truncated, info = env.step(np.zeros(7))

        if frames is not None:
            # If your obs contains images directly:
            if "front" in next_obs and "wrist" in next_obs:
                frames.append(
                    np.concatenate((next_obs["front"], next_obs["wrist"]), axis=0)
                )
            else:
                # Otherwise, render from the environment
                frame1, frame2 = env.render()
                frames.append(np.concatenate((frame1, frame2), axis=0))

        transition = dict(
            observations=copy.deepcopy(obs),
            actions=np.zeros(7),
            next_observations=copy.deepcopy(next_obs),
            rewards=rew,
            dones=done,
            infos=info
        )
        obs_list.append(transition["observations"])
        act_list.append(transition["actions"])
        next_obs_list.append(transition["next_observations"])
        rew_list.append(transition["rewards"])
        done_list.append(transition["dones"])
        info_list.append(transition["infos"])

        if done or truncated:
            return (obs_list, act_list, next_obs_list, rew_list, done_list, info_list), info.get("succeed", False)

        if (time.time() - start_wait_time) > WAIT_TIMEOUT:
            # Timed out waiting for Stage 1 final pose
            break

    # Extra few steps
    for _ in range(20):
        obs = next_obs
        next_obs, rew, done, truncated, info = env.step(np.zeros(7))
        transition = dict(
            observations=copy.deepcopy(obs),
            actions=np.zeros(7),
            next_observations=copy.deepcopy(next_obs),
            rewards=rew,
            dones=done,
            infos=info
        )
        if frames is not None:
            # If your obs contains images directly:
            if "front" in next_obs and "wrist" in next_obs:
                frames.append(
                    np.concatenate((next_obs["front"], next_obs["wrist"]), axis=0)
                )
            else:
                # Otherwise, render from the environment
                frame1, frame2 = env.render()
                frames.append(np.concatenate((frame1, frame2), axis=0))

        obs_list.append(transition["observations"])
        act_list.append(transition["actions"])
        next_obs_list.append(transition["next_observations"])
        rew_list.append(transition["rewards"])
        done_list.append(transition["dones"])
        info_list.append(transition["infos"])

        if done or truncated:
            return (obs_list, act_list, next_obs_list, rew_list, done_list, info_list), info.get("succeed", False)

    # --------------------------------------------------
    # 3) Stage 2: descend from above_port_pos to port_bottom_pos
    # --------------------------------------------------
    # Retrieve pose again (in case there's drift)
    initial_pos = d.sensor("connector_bottom_pos").data.copy()
    initial_quat = d.sensor("connector_head_quat").data.copy()

    port_bottom_pos = d.sensor("port_bottom_pos").data.copy()
    port_bottom_quat = d.sensor("port_bottom_quat").data.copy()

    N_stage2 = 10
    stage2_pos_waypoints, stage2_quat_waypoints = compute_waypoints(
        pos_start=initial_pos,
        quat_start=initial_quat,
        pos_end=port_bottom_pos,
        quat_end=port_bottom_quat,
        N=N_stage2
    )

    prev_pos = initial_pos
    prev_quat = initial_quat

    for i in range(N_stage2):
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

        obs = next_obs
        next_obs, rew, done, truncated, info = env.step(action)

        if frames is not None:
            # If your obs contains images directly:
            if "front" in next_obs and "wrist" in next_obs:
                frames.append(
                    np.concatenate((next_obs["front"], next_obs["wrist"]), axis=0)
                )
            else:
                # Otherwise, render from the environment
                frame1, frame2 = env.render()
                frames.append(np.concatenate((frame1, frame2), axis=0))

        transition = dict(
            observations=copy.deepcopy(obs),
            actions=copy.deepcopy(action),
            next_observations=copy.deepcopy(next_obs),
            rewards=rew,
            dones=done,
            infos=info
        )
        obs_list.append(transition["observations"])
        act_list.append(transition["actions"])
        next_obs_list.append(transition["next_observations"])
        rew_list.append(transition["rewards"])
        done_list.append(transition["dones"])
        info_list.append(transition["infos"])

        prev_pos = desired_pos
        prev_quat = desired_quat

        time.sleep(unwrapped_env.control_dt)

        if done or truncated:
            print("Environment ended prematurely during Stage 2.")
            print("info:", info)
            return (obs_list, act_list, next_obs_list, rew_list, done_list, info_list), info.get("succeed", False)

    # At this point we consider the episode finished. 
    # In your environment, you might rely on `done` to come from the environment automatically.
    # But if not done yet, do a few more steps or handle end conditions:
    for _ in range(50):
        obs = next_obs
        next_obs, rew, done, truncated, info = env.step(np.zeros(7))

        if frames is not None:
            # If your obs contains images directly:
            if "front" in next_obs and "wrist" in next_obs:
                frames.append(
                    np.concatenate((next_obs["front"], next_obs["wrist"]), axis=0)
                )
            else:
                # Otherwise, render from the environment
                frame1, frame2 = env.render()
                frames.append(np.concatenate((frame1, frame2), axis=0))

        transition = dict(
            observations=copy.deepcopy(obs),
            actions=np.zeros(7),
            next_observations=copy.deepcopy(next_obs),
            rewards=rew,
            dones=done,
            infos=info
        )
        obs_list.append(transition["observations"])
        act_list.append(transition["actions"])
        next_obs_list.append(transition["next_observations"])
        rew_list.append(transition["rewards"])
        done_list.append(transition["dones"])
        info_list.append(transition["infos"])
        if done or truncated:
            print("Environment ended prematurely during Stage 2.")
            print("info:", info)
            break

    # Return transitions for the entire episode + success flag
    return (obs_list, act_list, next_obs_list, rew_list, done_list, info_list), info.get("succeed", False)


def main(_):
    env = gymnasium.make("ur5ePegInHoleFixedGymEnv_state-v0", render_mode="human")
    unwrapped_env = env.unwrapped

    # Just to illustrate action & observation spaces
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    # We'll keep a list of ALL transitions from successful episodes only
    transitions = []

    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)

    # For storing frames if we need video
    frames = []
    episode_return = 0

    while success_count < success_needed:
        # Reset environment for a new episode
        obs, info = env.reset()
        episode_return = 0
        frames = []

        # Optionally record initial frame
        if FLAGS.save_video:
            if "front" in obs or "wrist" in obs:  # Check if image observations are available
                frames.append(np.concatenate((obs["front"], obs["wrist"]), axis=0))  # Combine views
            else:
                frame1, frame2 = env.render()
                frames.append(np.concatenate((frame1, frame2), axis=0))

        for _ in range(50):
            obs, rew, done, truncated, info = env.step(np.zeros(7))
        
            if frames is not None:
                # If your obs contains images directly:
                if "front" in obs and "wrist" in obs:
                    frames.append(
                        np.concatenate((obs["front"], obs["wrist"]), axis=0)
                    )
                else:
                    # Otherwise, render from the environment
                    frame1, frame2 = env.render()
                    frames.append(np.concatenate((frame1, frame2), axis=0))

        # ------------- Execute the 2-stage demo -------------
        (obs_list, act_list, next_obs_list, rew_list, done_list, info_list), success = run_two_stage_demo(env, unwrapped_env, frames=frames)


        # Compute return for logging
        episode_return = sum(rew_list)

        # If the environment provided a "succeed" flag, or you have some success metric:
        if success:
            pbar.set_description(f"Episode Return: {episode_return:.2f} [SUCCESS]")
            # Save these transitions
            for o, a, no, r, d, i in zip(obs_list, act_list, next_obs_list, rew_list, done_list, info_list):
                transitions.append(
                    dict(
                        observations=o,
                        actions=a,
                        next_observations=no,
                        rewards=r,
                        dones=d,
                        infos=i
                    )
                )
            success_count += 1
            pbar.update(1)

            # If saving video:
            if FLAGS.save_video:
                # Save frames from this successful run
                # We might want to gather frames inside the `run_two_stage_demo` as well
                # but here is the simplest approach:
                video_name = f"./demo_data/success_demo_{success_count}.mp4"
                imageio.mimsave(video_name, frames, fps=20)
                print(f"Saved video to {video_name}")
        else:
            pbar.set_description(f"Episode Return: {episode_return:.2f} [FAILED]")

    # Once we have all successful demos, save them
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

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
flags.DEFINE_boolean("save_video", False, "Flag to save videos of successful demos.")
flags.DEFINE_boolean("save_model", False, "Flag to save videos of successful demos.")



def compute_waypoints(pos_start, quat_start, pos_end, quat_end, resolution=0.01):
    """
    Returns position and orientation waypoints (with linear interpolation of position
    and slerp for orientation).
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


def run_two_stage_demo(env, unwrapped_env, _obs=None, trajectory=None, frames=None):
    """
    Example two-stage motion approach:
     1) Move from initial pose to an 'above port' pose
     2) Then move from that 'above port' pose down into the port

    Returns:
      - transitions: list of (obs, act, next_obs, reward, done, info) dicts
      - success: whether we ended with info["succeed"] == True
    """

    d = unwrapped_env.data
    obs = _obs

    # initial_pos = d.sensor("connector_bottom_pos").data.copy()
    # initial_quat = d.sensor("connector_head_quat").data.copy()

    initial_pos = d.mocap_pos[0].copy()
    initial_quat = d.mocap_quat[0].copy()
    port_bottom_pos = d.sensor("port_bottom_pos").data.copy()
    port_bottom_quat = d.sensor("port_bottom_quat").data.copy()

    # --------------------------------------------------
    # 1) Define an intermediate pose "above" the port
    # --------------------------------------------------
    random_offset_z = np.random.uniform(0.05, 0.1)
    above_port_pos = port_bottom_pos.copy()
    above_port_pos[2] += random_offset_z

    # --------------------------------------------------
    # 2) Compute Stage 1 waypoints
    # --------------------------------------------------
    stage1_pos_waypoints, stage1_quat_waypoints = compute_waypoints(
        pos_start=initial_pos,
        quat_start=initial_quat,
        pos_end=above_port_pos,
        quat_end=port_bottom_quat,
        resolution=0.001
    )

    # Execute Stage 1
    prev_pos = initial_pos
    prev_quat = initial_quat
    gripper_cmd = 0.0

    for i in range(len(stage1_pos_waypoints)):
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

        # Update for next iteration
        obs = next_obs  # or keep track of the last obs
        prev_pos = desired_pos
        prev_quat = desired_quat

        # Sleep until next control step
        time.sleep(unwrapped_env.control_dt)

        if done or truncated:
            # Episode ended prematurely
            return trajectory, info.get("succeed", False)

    # Wait until final pose is reached or until we exceed a timeout
    start_wait_time = time.time()
    WAIT_TIMEOUT = 10.0

    while np.linalg.norm(unwrapped_env.controller.error) > 1e-10:

        # Step in place with zero action while waiting
        next_obs, rew, done, truncated, info = env.step(np.zeros(7))
        frames.append(unwrapped_env.frames)

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=np.zeros(7),
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
            return trajectory, info.get("succeed", False)
        if (time.time() - start_wait_time) > WAIT_TIMEOUT:
            # Timed out waiting for Stage 1 final pose
            break

    # --------------------------------------------------
    # 3) Stage 2: descend from above_port_pos to port_bottom_pos
    # --------------------------------------------------
    # initial_pos = d.sensor("connector_bottom_pos").data.copy()
    # initial_quat = d.sensor("connector_head_quat").data.copy()
    
    initial_pos = d.mocap_pos[0].copy()
    initial_quat = d.mocap_quat[0].copy()
    port_bottom_pos = d.sensor("port_bottom_pos").data.copy()
    port_bottom_quat = d.sensor("port_bottom_quat").data.copy()

    stage2_pos_waypoints, stage2_quat_waypoints = compute_waypoints(
        pos_start=initial_pos,
        quat_start=initial_quat,
        pos_end=port_bottom_pos,
        quat_end=port_bottom_quat,
        resolution=0.001
    )

    prev_pos = initial_pos
    prev_quat = initial_quat

    for i in range(len(stage2_pos_waypoints)):
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

        next_obs, rew, done, truncated, info = env.step(action)
        frames.append(unwrapped_env.frames)

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=action,
                next_observations=next_obs,
                rewards=rew,               
                masks=1.0 - done,
                dones=done,
                infos=info
            )
        )
        trajectory.append(transition)
        
        obs = next_obs
        prev_pos = desired_pos
        prev_quat = desired_quat

        time.sleep(unwrapped_env.control_dt)

        if done or truncated:
            print("Environment done during Stage 2.")
            return trajectory, info.get("succeed", False)
        
    # At this point we consider the episode finished. 
    # In your environment, you might rely on `done` to come from the environment automatically.
    # But if not done yet, do a few more steps or handle end conditions:
    start_wait_time = time.time()

    while np.linalg.norm(unwrapped_env.controller.error) > 1e-10:
        next_obs, rew, done, truncated, info = env.step(np.zeros(7))
        frames.append(unwrapped_env.frames)

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=np.zeros(7),
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
            print("Environment done during Stage 2.")
            break
        if (time.time() - start_wait_time) > WAIT_TIMEOUT:
            # Timed out waiting for Stage 1 final pose
            print("Environment ended prematurely during Stage 2.(Timeout)")
            break

    # Return transitions for the entire episode + success flag
    return trajectory, info.get("succeed", False)


def main(_):
    env = gymnasium.make("ur5ePegInHoleFixedGymEnv_state-v0", render_mode="human")
    unwrapped_env = env.unwrapped

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

        while np.linalg.norm(unwrapped_env.controller.error) > 1e-10:
            next_obs, rew, done, truncated, info = env.step(np.zeros(7)) 
            frames.append(unwrapped_env.frames)
            transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=np.zeros(7),
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
            )
            trajectory.append(transition)
            obs = next_obs

        # ------------- Execute the 2-stage demo -------------
        trajectory, success = run_two_stage_demo(env, unwrapped_env, _obs=obs, trajectory=trajectory, frames=frames)

        # Compute episode return (sum of rewards in 'trajectory')
        returns = sum(t["rewards"] for t in trajectory)

        # If the environment provided a "succeed" flag, or you have some success metric:
        if success:
            pbar.set_description(f"Episode Return: {returns:.2f} [SUCCESS]")
            success_count += 1
            pbar.update(1)

            # Add the entire trajectory to the global transitions list
            for t in trajectory:
                transitions.append(copy.deepcopy(t))

            # If saving video:
            if FLAGS.save_video:
                # Save frames from this successful run
                # We might want to gather frames inside the `run_two_stage_demo` as well
                # but here is the simplest approach:
                video_name = f"./demo_data/success_demo_{success_count}.mp4"
                imageio.mimsave(video_name, frames, fps=20)
                print(f"Saved video to {video_name}")
        else:
            pbar.set_description(f"Episode Return: {returns:.2f} [FAILED]")
        trajectory = []
        frames = []  # Clear frames for the next episode
        returns = 0
        obs, info = env.reset()
        
    # Once we have all successful demos, save them
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

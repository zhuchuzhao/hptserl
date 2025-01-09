import numpy as np
import time
import gymnasium
import mujoco
import mujoco.viewer
import numpy as np
import mujoco_sim

# def move_to_position(env, target_pos, target_quat=None, 
#                      tolerance=0.001, max_steps=500):
#     """
#     Moves the end-effector to a specified position (and orientation),
#     stepping the simulation until close enough or max_steps reached.
#     """
#     for i in range(max_steps):
#         # Update mocap position
#         env.data.mocap_pos[0] = target_pos

#         # Optionally update orientation
#         if target_quat is not None:
#             env.data.mocap_quat[0] = target_quat

#         # Step the environment
#         obs, rew, done, truncated, info = env.step(np.zeros(7))

#         # Check distance
#         current_pos = env.data.sensor("hande/pinch_pos").data
#         dist = np.linalg.norm(current_pos - target_pos)
#         if dist < tolerance:

#             print(f"Reached target: {target_pos} (within tolerance {tolerance})")
#             break

def move_to_position(env, target_pos, target_quat=None, 
                     tolerance=0.001, max_steps=500, resolution=1):
    """
    Moves the end-effector to a specified position (and orientation),
    by calculating error and generating actions based on it.
    Stepping the simulation until close enough or max_steps reached.
    """
    for i in range(max_steps):
        # Get the current position and orientation from the environment
        current_pos = env.unwrapped.data.sensor("hande/pinch_pos").data
        current_quat = env.unwrapped.data.sensor("hande/pinch_quat").data

        # Compute position error
        pos_error = target_pos - current_pos
        print(f"Position error: {np.linalg.norm(pos_error)}")

        # Generate a positional action with a resolution limit
        pos_action = np.clip(pos_error, -resolution, resolution)
        print(f"Position action: {pos_action}")


        # # Compute orientation error and generate orientation action
        # if target_quat is not None:
        #     quat_conj = np.zeros(4)
        #     quat_err = np.zeros(4)
        #     mujoco.mju_negQuat(quat_conj, current_quat)
        #     mujoco.mju_mulQuat(quat_err, target_quat, quat_conj)

        #     # Convert quaternion error to Euler angle error
        #     ori_error = np.zeros(3)
        #     mujoco.mju_quat2Vel(ori_error, quat_err, 1.0)

        #     # Generate orientation action with a resolution limit
        #     ori_action = np.clip(ori_error, -resolution, resolution)
        # else:
        ori_action = np.zeros(3)

        # Combine position and orientation actions, add a zero gripper action
        action = np.concatenate((pos_action, ori_action, [0.0]))
        print(f"Action: {action}")

        # Step the environment with the generated action
        obs, rew, done, truncated, info = env.step(action)

        # Check if the positional error is within tolerance
        if np.linalg.norm(pos_error) < tolerance:
            print(f"Reached target: {target_pos} (within tolerance {tolerance})")
            break

        # # Optional: Check orientation tolerance
        # if target_quat is not None and np.linalg.norm(ori_error) < tolerance:
        #     print(f"Reached target orientation: {target_quat} (within tolerance {tolerance})")
        #     break

    else:
        print(f"Failed to reach target position within {max_steps} steps.")


def pick_and_place_connector(env):
    """
    High-level routine to pick the connector and place it in the port.
    """
    # 1. Reset environment
    

    # 6. Move above the port
    port_bottom_pos = env.unwrapped.data.sensor("port_bottom_pos").data.copy()
    port_bottom_quat = env.unwrapped.data.sensor("port_bottom_quat").data
    pos_above_port = port_bottom_pos.copy()
    pos_above_port[2] += 2
    move_to_position(env, pos_above_port, tolerance=0.001)

    # # 8. Move down to insert connector
    # pos_in_port = env.unwrapped.data.site_xpos[env.unwrapped._port_site_id][:3]
    # move_to_position(env, pos_in_port, port_bottom_quat, tolerance=0.001)

    print("Pick-and-place routine complete!")


if __name__ == "__main__":
    env = gymnasium.make("ur5ePegInHoleFixedGymEnv_vision-v0", render_mode="human")
    # Unwrapping the environment
    unwrapped_env = env.unwrapped
    m = unwrapped_env.model
    d = unwrapped_env.data

    env.reset()

    # 6. Move above the port
    port_bottom_pos = d.sensor("port_bottom_pos").data.copy()
    port_bottom_pos[2] += 0.1

    max_steps = 500
    tolerance = 0.001
    resolution = 0.001125
    for i in range(50):
        obs, rew, done, truncated, info = env.step(np.zeros(7))
    while True:

        # Get the current position and orientation from the environment
        current_pos = d.sensor("connector_bottom_pos").data.copy()
        pos_error = port_bottom_pos - current_pos
        print("Position error:", pos_error)
        print(f"Position error: {np.linalg.norm(pos_error)}")
        pos_action = np.clip(pos_error, -resolution, resolution)
        action = np.concatenate((pos_action, np.zeros(3), [0.0]))
        print(f"Action: {action}")
        obs, rew, done, truncated, info = env.step(action)
        if np.linalg.norm(pos_error) < tolerance:
            print(f"Reached target: {port_bottom_pos} (within tolerance {tolerance})")
            break

    while True:
        # Get the current position and orientation from the environment
        current_pos = d.sensor("connector_bottom_pos").data.copy()
        pos_error = port_bottom_pos - current_pos
        print("Position error:", pos_error)
        print(f"Position error: {np.linalg.norm(pos_error)}")
        pos_action = np.clip(pos_error, -resolution, resolution)
        action = np.concatenate((pos_action, np.zeros(3), [0.0]))
        print(f"Action: {action}")
        obs, rew, done, truncated, info = env.step(action)
        if np.linalg.norm(pos_error) < tolerance:
            print(f"Reached target: {port_bottom_pos} (within tolerance {tolerance})")
            break

    print("Environment will continue running. Press CTRL+C to exit.")
    while True:
        # For instance, just send zero actions indefinitely
        obs, rew, done, truncated, info = env.step(np.zeros(7))
        if done or truncated:
            obs, info = env.reset()
    env.close()

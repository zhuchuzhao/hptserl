import numpy as np
import time
import gymnasium
import mujoco
import mujoco.viewer
import numpy as np
import mujoco_sim

def move_to_position(env, target_pos, target_quat=None, 
                     tolerance=0.001, max_steps=500):
    """
    Moves the end-effector to a specified position (and orientation),
    stepping the simulation until close enough or max_steps reached.
    """
    for i in range(max_steps):
        # Update mocap position
        env.data.mocap_pos[0] = target_pos

        # Optionally update orientation
        if target_quat is not None:
            env.data.mocap_quat[0] = target_quat

        # Step the environment
        obs, rew, done, truncated, info = env.step(np.zeros(7))

        # Check distance
        current_pos = env.data.sensor("hande/pinch_pos").data
        dist = np.linalg.norm(current_pos - target_pos)
        if dist < tolerance:
            print(f"Reached target: {target_pos} (within tolerance {tolerance})")
            break


def close_gripper(env, close_amount=1.0):
    """
    Closes the gripper by setting the gripper actuator control to
    a desired fraction (0.0 to 1.0).
    """
    # Scale from [0.0..1.0] to [0..255]
    env.data.ctrl[env._gripper_ctrl_id] = close_amount * 255.0
    # Step a few times to finalize closure
    for _ in range(20):
        env.step(np.zeros(7))

def open_gripper(env):
    """
    Opens the gripper.
    """
    env.data.ctrl[env._gripper_ctrl_id] = 0.0
    for _ in range(20):
        env.step(np.zeros(7))


def pick_and_place_connector(env):
    """
    High-level routine to pick the connector and place it in the port.
    """
    # 1. Reset environment
    obs, info = env.reset()

    # 2. Move above the connector
    connector_head_pos = env.data.sensor("connector_head_pos").data.copy()
    pos_above_connector = connector_head_pos.copy()
    pos_above_connector[2] += 0.10  # hover 10cm above
    move_to_position(env, pos_above_connector)

    # 3. Move down onto the connector
    #    Just enough to 'touch' or be very close
    pos_on_connector = connector_head_pos.copy()
    pos_on_connector[2] += 0.01 
    move_to_position(env, pos_on_connector, tolerance=0.001)

    # 4. Close the gripper to grab the connector
    close_gripper(env, close_amount=1.0)

    # 5. Lift the connector
    pos_lifted = pos_on_connector.copy()
    pos_lifted[2] += 0.10
    move_to_position(env, pos_lifted)

    # 6. Move above the port
    # port_bottom_pos = env.data.sensor("port_bottom_pos").data.copy()
    port_bottom_pos = env.data.site_xpos[env._port_site_id][:3]
    pos_above_port = port_bottom_pos.copy()
    pos_above_port[2] += 0.10
    move_to_position(env, pos_above_port, tolerance=0.001)

    # 7. Align vertically (optional: set a desired orientation)
    # If the port is oriented a certain way, you can set `target_quat`.
    # For demonstration, we assume the current mocap orientation is fine.

    # 8. Move down to insert connector
    pos_in_port = env.data.site_xpos[env._port_site_id][:3]
    move_to_position(env, pos_in_port, tolerance=0.001)

    # # 9. Release the connector
    # open_gripper(env)

    # # 10. (Optional) Move away
    # pos_away = pos_in_port.copy()
    # pos_away[2] += 0.10
    # move_to_position(env, pos_away)

    print("Pick-and-place routine complete!")


if __name__ == "__main__":
    env = gymnasium.make("ur5ePegInHoleGymEnv_state-v0", render_mode="human")
    env.reset()
    pick_and_place_connector(env.unwrapped)
    print("Environment will continue running. Press CTRL+C to exit.")
    while True:
        # For instance, just send zero actions indefinitely
        obs, rew, done, truncated, info = env.step(np.zeros(7))
        if done or truncated:
            obs, info = env.reset()
    env.close()

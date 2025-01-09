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


def pick_and_place_connector(env):
    """
    High-level routine to pick the connector and place it in the port.
    """
    # 1. Reset environment
    obs, info = env.reset()

    # 6. Move above the port
    port_bottom_pos = env.data.sensor("port_bottom_pos").data.copy()
    port_bottom_quat = env.data.sensor("port_bottom_quat").data
    pos_above_port = port_bottom_pos.copy()
    pos_above_port[2] += 0.10
    move_to_position(env, pos_above_port, tolerance=0.001)

    # 8. Move down to insert connector
    pos_in_port = env.data.site_xpos[env._port_site_id][:3]
    move_to_position(env, pos_in_port, port_bottom_quat, tolerance=0.001)


    print("Pick-and-place routine complete!")


if __name__ == "__main__":
    env = gymnasium.make("ur5ePegInHoleFixedGymEnv_vision-v0", render_mode="human")
    env.reset()
    pick_and_place_connector(env.unwrapped)
    print("Environment will continue running. Press CTRL+C to exit.")
    while True:
        # For instance, just send zero actions indefinitely
        obs, rew, done, truncated, info = env.step(np.zeros(7))
        if done or truncated:
            obs, info = env.reset()
    env.close()

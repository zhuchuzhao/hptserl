import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import gymnasium
import mujoco_sim
import imageio


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 1, "Number of successful demos to collect.")
flags.DEFINE_boolean("save_video", True, "Flag to save videos of successful demos.")

def main(_):

    env = gymnasium.make("ur5ePegInHoleFixedGymEnv_state-v0", render_mode="human")
    action_spec = env.action_space
    print(f"Action space: {action_spec}")

    observation_spec = env.observation_space
    print(f"Observation space: {observation_spec}")
    
    obs, info = env.reset()
    print("obs", obs)
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    frames = []  # For storing video frames
    returns = 0
    
    while success_count < success_needed:
        if FLAGS.save_video:  # Only collect frames if save_video flag is active
                    if "front" in obs or "wrist" in obs:  # Check if image observations are available
                        frames.append(np.concatenate((obs["front"], obs["wrist"]), axis=0))  # Combine views
                    else:
                        frame1, frame2 = env.render()
                        print(frame1.shape, frame2.shape)
                        frames.append(np.concatenate((frame1, frame2), axis=0))

        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done:
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
                # Save the video for the successful demo if save_video flag is active
                if FLAGS.save_video:
                    # Save the video for the successful demo
                    video_name = f"./demo_data/success_demo_{success_count}.mp4"
                    imageio.mimsave(video_name, frames, fps=20)
                    print(f"Saved video to {video_name}")
            trajectory = []
            frames = []  # Clear frames for the next episode
            returns = 0
            obs, info = env.reset()
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)
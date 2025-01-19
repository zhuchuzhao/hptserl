#!/usr/bin/env python3
import os
import torch
import pickle as pkl
import numpy as np

def recursive_convert(x):
    """
    Recursively convert torch.Tensor objects.
    
    • If a tensor is scalar (has no dimensions or only one element), return a plain Python scalar.
    • Otherwise, convert to a NumPy array.
    • Recurse into dictionaries and lists.
    """
    if hasattr(x, "detach"):  # likely a torch.Tensor
        if x.ndim == 0 or x.numel() == 1:
            return x.item()  # return plain Python number
        else:
            return x.detach().cpu().numpy()
    elif isinstance(x, dict):
        return {k: recursive_convert(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(recursive_convert(item) for item in x)
    return x

def load_pt_as_transitions(pt_file):
    data = torch.load(pt_file, map_location="cpu")
    transitions_list = []
    
    # Iterate over episodes (top-level keys)
    for ep_key, episode in data.items():
        # For each episode, iterate over its steps
        for step_key, trans_data in episode.items():
            if isinstance(trans_data, dict) and "observation" in trans_data:
                new_trans = {}
                # --- Remap keys to match the original 20_demos.pkl format ---
                # In 20_demos.pkl, observations is a dictionary with a key "state"
                new_trans["observations"] = {"state": trans_data["observation"]}
                new_trans["next_observations"] = {"state": trans_data["next_observation"]}
                
                # Use plural "actions" to match original demos
                new_trans["actions"] = trans_data["action"]
                new_trans["rewards"] = trans_data["reward"]
                # Rename 'info' to 'infos' (if no further mapping is needed)
                new_trans["infos"] = trans_data["info"]
                
                # Create done flag and mask from terminated and truncated.
                terminated = trans_data["terminated"]
                truncated = trans_data["truncated"]
                if hasattr(terminated, "item"):
                    terminated = terminated.item()
                if hasattr(truncated, "item"):
                    truncated = truncated.item()
                done = bool(terminated or truncated)
                new_trans["dones"] = done
                new_trans["masks"] = 0.0 if done else 1.0

                # (Optional) Process actions if you need to slice them.
                # For instance, if you expect actions to be of dimension 7:
                action_array = new_trans["actions"]
                if hasattr(action_array, "numpy"):
                    action_array = action_array.numpy()
                # Uncomment and adjust the slicing if necessary:
                # if np.linalg.norm(action_array) > 0.0 and action_array.shape[0] > 6:
                #     new_trans["actions"] = trans_data["action"][:7]
                
                # --- Convert all fields recursively ---
                new_trans = {k: recursive_convert(v) for k, v in new_trans.items()}
                transitions_list.append(new_trans)
    return transitions_list

if __name__ == "__main__":
    pt_file = os.path.join(os.getcwd(), "demo_256_episodes_v6.pt")
    transitions = load_pt_as_transitions(pt_file)
    print(f"Converted {len(transitions)} transitions from {pt_file}")
    
    # Inspect the first transition’s structure.
    sample = transitions[0]
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"{key}: dict with keys: {list(value.keys())}")
        elif hasattr(value, "shape"):
            print(f"{key}: {type(value).__name__}, shape: {value.shape}")
        else:
            print(f"{key}: {type(value).__name__}, value: {value}")
    
    converted_file = os.path.join(os.getcwd(), "converted_demo.pkl")
    with open(converted_file, "wb") as f:
        pkl.dump(transitions, f)
    print(f"Saved converted transitions to {converted_file}")

#!/usr/bin/env python3
import argparse
import os
import torch
import pickle

def load_file(file_path):
    """
    Attempt to load a file using torch.load first; if that fails,
    attempt to load using pickle.load.
    """
    try:
        data = torch.load(file_path, map_location="cpu")
        print("Loaded file using torch.load\n")
        return data
    except Exception as e:
        print(f"torch.load failed: {e}")
        print("Attempting to load using pickle...\n")
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print("Loaded file using pickle.load\n")
        return data
    except Exception as e:
        print(f"pickle.load failed: {e}")
    return None

def summarize_structure(data, indent=0, current_depth=0, max_depth=3):
    """
    Recursively summarizes the structure of nested objects (dictionaries,
    lists, tensors, or any object with a shape attribute) up to a maximum depth.
    It prints the type, keys (if applicable), and shape (if present).
    """
    prefix = " " * indent

    # Stop if we've reached the maximum depth.
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return

    if isinstance(data, dict):
        print(f"{prefix}Dictionary with {len(data)} keys. Keys: {list(data.keys())}")
        # Go one level deeper for each key
        for key, value in data.items():
            # Show the type and (if available) shape
            if hasattr(value, "shape"):
                try:
                    shape = value.shape
                except Exception:
                    shape = "unknown"
                print(f"{prefix}  {key} -> {type(value).__name__}, shape: {shape}")
            else:
                print(f"{prefix}  {key} -> {type(value).__name__}")
            # Only further summarize if the value is itself a dict or list.
            if isinstance(value, (dict, list)):
                summarize_structure(value, indent=indent+4, current_depth=current_depth+1, max_depth=max_depth)
    elif isinstance(data, list):
        print(f"{prefix}List with {len(data)} items. [Showing structure of first item:]")
        if len(data) > 0:
            summarize_structure(data[0], indent=indent+4, current_depth=current_depth+1, max_depth=max_depth)
    elif isinstance(data, torch.Tensor):
        print(f"{prefix}Tensor, shape: {tuple(data.shape)}, dtype: {data.dtype}")
    elif hasattr(data, "shape"):
        # This covers objects such as numpy arrays
        try:
            shape = data.shape
        except Exception:
            shape = "unknown"
        print(f"{prefix}{type(data).__name__}, shape: {shape}")
    else:
        # For other types, just show the type.
        print(f"{prefix}{type(data).__name__}")

def examine_file(file_path, deep=False):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    data = load_file(file_path)
    if data is None:
        print("Failed to load data.")
        return

    print(f"Successfully loaded: {file_path}\n")
    
    # Top-level summary
    if isinstance(data, dict):
        print("File is a dictionary. Top-level keys:")
        for key, value in data.items():
            if hasattr(value, "shape"):
                try:
                    shape = value.shape
                except Exception:
                    shape = "unknown"
                print(f" - {key} -> {type(value).__name__}, shape: {shape}")
            else:
                print(f" - {key} -> {type(value).__name__}")
        # If deep mode is enabled, summarize nested structure for keys starting with "episode_"
        if deep:
            print("\nSummarizing nested structure for keys that appear to be episodes:")
            for key, value in data.items():
                if key.startswith("episode_") and isinstance(value, dict):
                    print(f"\n{key} -> Dictionary with keys: {list(value.keys())}")
                    summarize_structure(value, indent=4, current_depth=0, max_depth=3)
    elif isinstance(data, list):
        print(f"File is a list with {len(data)} items.")
        summarize_structure(data, indent=2, current_depth=0, max_depth=3)
    elif isinstance(data, torch.Tensor):
        print(f"File is a Tensor, shape: {tuple(data.shape)}, dtype: {data.dtype}")
    elif hasattr(data, "shape"):
        try:
            shape = data.shape
        except Exception:
            shape = "unknown"
        print(f"File is an object of type {type(data).__name__}, shape: {shape}")
    else:
        print(f"File is an object of type: {type(data).__name__}")

def main():
    parser = argparse.ArgumentParser(
        description="Summarize the structure of a file (.pth, .pt, or .pkl) by printing types, keys, and shapes (if available) up to 2 levels deep."
    )
    parser.add_argument("file_path", help="Path to the file")
    parser.add_argument(
        "--deep", action="store_true",
        help="Examine nested structure for keys like 'episode_' (up to 2 levels deep)."
    )
    args = parser.parse_args()
    examine_file(file_path=args.file_path, deep=args.deep)

if __name__ == "__main__":
    main()

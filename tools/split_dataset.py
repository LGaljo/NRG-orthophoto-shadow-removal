#!/usr/bin/env python
"""
Split Dataset Script

This script creates a smaller dataset by randomly selecting a percentage of files
from each subdirectory of the source dataset and copying them to a new destination
directory while preserving the original folder structure.

The script uses hardcoded values:
- Source directory: dataset\\unity_dataset\\mixed_visibility_dataset\\train
- Destination directory: dataset\\unity_dataset\\mixed_visibility_dataset\\train_small
- Percentage of files to select: 25%
- Random seed: 42

Notes:
    - The script preserves the original folder structure in the destination directory
    - 25% of the files are randomly selected from each subdirectory
    - The script uses a random seed (42) for reproducibility
"""

import os
import random
import shutil
from pathlib import Path


def split_dataset(source_dir, dest_dir, percentage=0.25, seed=42):
    """
    Split a dataset by randomly selecting a percentage of files from each subdirectory
    and copying them to a new destination folder while preserving the original structure.
    
    Args:
        source_dir (str): Path to the source dataset directory
        dest_dir (str): Path to the destination directory where the split dataset will be created
        percentage (float): Percentage of files to select (between 0 and 1)
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    # random.seed(seed)
    
    # Convert paths to Path objects for better cross-platform compatibility
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Check if source directory exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist")
    
    # Create destination directory if it doesn't exist
    if not dest_path.exists():
        os.makedirs(dest_path)
        print(f"Created destination directory: {dest_dir}")
    
    # Get all subdirectories in the source directory
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No subdirectories found in {source_dir}")
        return
    
    # Create new subdirectories
    for subdir in subdirs:
        subdir_name = subdir.name
        dest_subdir = dest_path / subdir_name
        
        # Create corresponding subdirectory in destination
        if not dest_subdir.exists():
            os.makedirs(dest_subdir)
            print(f"Created subdirectory: {dest_subdir}")


    # Get all files in the current subdirectory
    files = [f for f in (source_path / "train_A").iterdir() if f.is_file()]

    # Calculate how many files to select
    num_files_to_select = max(1, int(len(files) * percentage))

    # Randomly select files
    selected_files = random.sample(files, num_files_to_select)

    # Copy selected files to destination
    for file in selected_files:
        dest_file = dest_path / "train_A" / file.name
        shutil.copy2(file, dest_file)
        file = source_path / "train_B" / file.name.replace("Hard", "Mask")
        dest_file = dest_path / "train_B" / file.name.replace("Hard", "Mask")
        shutil.copy2(file, dest_file)
        file = source_path / "train_C" / file.name.replace("Mask", "None")
        dest_file = dest_path / "train_C" / file.name.replace("Mask", "None")
        shutil.copy2(file, dest_file)

    print(f"Copied {len(selected_files)} files ({percentage*100:.1f}% of {len(files)} files)")

    print(f"\nDataset split complete. {percentage*100:.1f}% of the original dataset copied to {dest_dir}")


if __name__ == "__main__":
    # Hardcoded values
    source_dir = "..\\dataset\\unity_dataset\\usos\\train"
    dest_dir = "..\\dataset\\unity_dataset\\usos_xs\\train"
    percentage = 0.01
    seed = 42
    
    # Call the split_dataset function with hardcoded values
    split_dataset(source_dir, dest_dir, percentage, seed)

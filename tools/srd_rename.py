import os
import sys
import re

def rename_files():
    """
    Rename files in train_C folder by removing the '_free' suffix to match files in train_A folder.
    """
    # Path to the dataset directory
    base_path = os.path.join('st_cgan', 'trained_models', 'st-cgan', 'srd_results', 'dataset', 'test')
    
    # Full paths to train_A and train_C folders
    train_c_path = os.path.join(base_path, 'train_C')
    
    # Check if the directories exist
    if not os.path.exists(train_c_path):
        print(f"Error: Directory {train_c_path} does not exist.")
        return
    
    # Count for statistics
    renamed_count = 0
    error_count = 0
    
    # Process files in train_C
    for filename in os.listdir(train_c_path):
        if '_free' in filename:
            # Remove '_free' from the filename
            new_filename = filename.replace('_free', '')
            
            # Full paths for the old and new filenames
            old_file_path = os.path.join(train_c_path, filename)
            new_file_path = os.path.join(train_c_path, new_filename)
            
            try:
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {filename}: {str(e)}")
                error_count += 1
    
    # Print summary
    print(f"\nRenaming complete!")
    print(f"Total files renamed: {renamed_count}")
    print(f"Errors encountered: {error_count}")

if __name__ == "__main__":
    rename_files()

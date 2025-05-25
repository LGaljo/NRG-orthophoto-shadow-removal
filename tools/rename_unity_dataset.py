import re
from pathlib import Path

# Root dataset path
base_dir = Path("../dataset/unity_dataset/mixed_visibility_dataset/test - Copy")

# Folders and suffixes to strip per folder
folder_suffix_map = {
    "train_A": r"-(Hard|Soft|Medium)-(\d+\.png)$",
    "train_B": r"-Mask-(\d+\.png)$",
    "train_C": r"-None-(\d+\.png)$",
}

def rename_files(folder_name, pattern_str):
    folder_path = base_dir / folder_name
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return

    pattern = re.compile(rf"(.*){pattern_str}")

    for file in folder_path.iterdir():
        if file.is_file() and file.suffix == '.png':
            match = pattern.match(file.name)
            if match:
                base = match.group(1)
                index = match.group(len(match.groups()))
                new_name = f"{base}-{index}"
                new_path = folder_path / new_name
                print(f"[{folder_name}] Renaming: {file.name} -> {new_name}")
                file.rename(new_path)

def main():
    for folder, suffix_pattern in folder_suffix_map.items():
        rename_files(folder, suffix_pattern)

if __name__ == "__main__":
    main()

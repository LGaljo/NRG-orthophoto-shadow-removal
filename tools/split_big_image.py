import glob
import os.path
import re
import concurrent.futures
import shutil

import numpy as np
import torch
from PIL import Image
from imutils import paths
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# subset_path = "../unity_dataset/many_houses_dataset/full_size/DOF5-20240602-D0733*.png"
fullsize_path = "C:\\Users\\lukag\\AppData\\LocalLow\\Magistrska naloge - Luka Galjot\\Mag Generate Shadows/*.png"
# subset_path = "C:\\Users\\lukag\\AppData\\LocalLow\\Magistrska naloge - Luka Galjot\\Mag Generate Shadows\\*.png"

create_mask = False
with_mp = True

test_size = 0.1

dataset_root = '../dataset/unity_dataset/mixed_visibility_dataset_320/'
image_train = 'train_A'
image_mask = 'train_B'
image_ground_truth = 'train_C'

pattern = r".*[-]([x]\d+)[-]([z]\d+)[-](\w*).*"
# pattern = r"[-](.*)[-]([x]+\d+).*([z]+\d+).*[.]+.*"
# pattern = r"^.*_([a-z0-9]+){1}-(x\d+)+-(z\d+)+"

tile_size = 320
resize_to = 320

"""
Split large image image into tiles.

Set tile_size var to define the size of each tile
"""
def process_image(image_path):
    match = re.search(pattern, image_path)
    type = match.group(3)

    if type == 'Mask':
        # For mask images
        image = Image.open(image_path).convert('L')
    else:
        # For normal and shadowless images
        image = Image.open(image_path).convert('RGB')

    # Convert image to tensor
    image_tensor = torch.tensor(np.array(image))

    # Separate image tensor to tiles of "tile_size" resolution
    tiles = image_tensor.unfold(0, tile_size, tile_size).unfold(1, tile_size, tile_size)

    idx = 1
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            tile = tiles[i, j]
            filename = os.path.basename(image_path).split('.')[0]
            folder_path = None

            if type == 'Mask' and create_mask:
                folder_path = image_mask
            if type == 'None':
                folder_path = image_ground_truth
            if type == 'Hard':
                folder_path = image_train

            # If image is RGB
            if tile.shape[0] == 3:
                tile = tile.permute(1, 2, 0).contiguous().view(tile_size, tile_size, 3)

            save_image(tile, dataset_root + folder_path, filename, idx)
            idx += 1


def save_image(img, fp, fn, idx):
    img = img.squeeze()
    if img.shape[0] == 3 or img.shape[1] == 3 or img.shape[2] == 3:
        img = Image.fromarray(img.numpy().astype(np.uint8), 'RGB')
    else:
        img = Image.fromarray(img.numpy().astype(np.uint8), 'L')

    if tile_size != resize_to:
        img = img.resize((resize_to, resize_to))

    img.save(f"{fp}/{fn}-{idx}.png", "PNG")


def split_train_test():
    image_t = sorted(list(paths.list_images(f"{dataset_root}/{image_train}")))
    image_gt = sorted(list(paths.list_images(f"{dataset_root}/{image_ground_truth}")))
    image_m = sorted(list(paths.list_images(f"{dataset_root}/{image_mask}")))

    train_t, test_t, train_gt, test_gt, train_m, test_m = [], [], [], [], [], []
    if len(image_m) > 0:
        assert len(image_t) == len(image_gt) == len(image_m)
        train_t, test_t, train_gt, test_gt, train_m, test_m = train_test_split(image_t, image_gt, image_m, test_size=0.1)
    else:
        assert len(image_t) == len(image_gt)
        train_t, test_t, train_gt, test_gt = train_test_split(image_t, image_gt, test_size=0.1)

    # Define directories for the train/test split
    train_dir = f"{dataset_root}/train"
    test_dir = f"{dataset_root}/test"

    # Create the necessary folders if they don't exist
    os.makedirs(f"{train_dir}/{image_train}", exist_ok=True)
    os.makedirs(f"{test_dir}/{image_train}", exist_ok=True)

    os.makedirs(f"{train_dir}/{image_ground_truth}", exist_ok=True)
    os.makedirs(f"{test_dir}/{image_ground_truth}", exist_ok=True)

    if len(image_m) > 0:
        os.makedirs(f"{train_dir}/{image_mask}", exist_ok=True)
        os.makedirs(f"{test_dir}/{image_mask}", exist_ok=True)

    # Function to move images to the corresponding folders
    def move_images(image_list, target_dir):
        for img_path in image_list:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(target_dir, filename)
            shutil.move(str(img_path), str(dest_path))

    # Move images
    move_images(train_t, f"{train_dir}/{image_train}")
    move_images(test_t, f"{test_dir}/{image_train}")

    move_images(train_gt, f"{train_dir}/{image_ground_truth}")
    move_images(test_gt, f"{test_dir}/{image_ground_truth}")

    if len(image_m) > 0:
        move_images(train_m, f"{train_dir}/{image_mask}")
        move_images(test_m, f"{test_dir}/{image_mask}")

    print("Images have been successfully split and moved.")




if __name__ == '__main__':
    if not os.path.exists(dataset_root):
        os.mkdir(dataset_root)
    if not os.path.exists(dataset_root + image_train):
        os.mkdir(dataset_root + image_train)
    if not os.path.exists(dataset_root + image_mask) and create_mask:
        os.mkdir(dataset_root + image_mask)
    if not os.path.exists(dataset_root + image_ground_truth):
        os.mkdir(dataset_root + image_ground_truth)

    subset_images = glob.glob(fullsize_path)

    if with_mp:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_image, subset_images), total=len(subset_images)))
    else:
        for image in tqdm(subset_images, total=len(subset_images)):
            process_image(image)

    split_train_test()
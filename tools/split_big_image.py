import glob
import os.path
import re
import concurrent.futures
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

subset_path = "../unity_dataset/big_trees_datasets/full_size/*.png"
# subset_path = "C:\\Users\\lukag\\AppData\\LocalLow\\Magistrska naloge - Luka Galjot\\Mag Generate Shadows\\*.png"

image_normal = '../unity_dataset/big_trees_datasets/train_A'
image_mask = '../unity_dataset/big_trees_datasets/train_B'
image_shadowless = '../unity_dataset/big_trees_datasets/train_C'

pattern = r"[-](.*)[-]([x]+\d+).*([z]+\d+).*[.]+.*"
# pattern = r"^.*_([a-z0-9]+){1}-(x\d+)+-(z\d+)+"

# tile_size = 256
tile_size = 850
resize_to = 320

"""
Split large image image into tiles.

Set tile_size var to define the size of each tile
"""


def process_image(image_path):
    if 'Mask' in image_path:
        # For mask images
        image = Image.open(image_path).convert('L')
    else:
        # For normal and shadowless images
        image = Image.open(image_path).convert('RGB')
    image_tensor = torch.tensor(np.array(image))
    tiles = image_tensor.unfold(0, tile_size, tile_size).unfold(1, tile_size, tile_size)

    idx = 1
    match = re.search(pattern, image_path)
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            tile = tiles[i, j]
            file_name = "{}_{}_{}".format(match.group(1), match.group(2), match.group(3))
            if 'Mask' in image_path:
                tile_image = Image.fromarray(tile.numpy().astype(np.uint8), 'L')
                folder_path = image_mask
            elif 'None' in image_path:
                reshaped_image = tile.permute(1, 2, 0).contiguous().view(tile_size, tile_size, 3)
                tile_image = Image.fromarray(reshaped_image.numpy().astype(np.uint8), 'RGB')
                folder_path = image_shadowless
            else:
                reshaped_image = tile.permute(1, 2, 0).contiguous().view(tile_size, tile_size, 3)
                tile_image = Image.fromarray(reshaped_image.numpy().astype(np.uint8), 'RGB')
                folder_path = image_normal
            if tile_size != resize_to:
                tile_image = tile_image.resize((resize_to, resize_to))
            tile_image.save(folder_path + f"/{file_name}-{idx}.png", "PNG")
            idx += 1


if __name__ == '__main__':
    if not os.path.exists(image_normal):
        os.mkdir(image_normal)
    if not os.path.exists(image_mask):
        os.mkdir(image_mask)
    if not os.path.exists(image_shadowless):
        os.mkdir(image_shadowless)

    subset_images = glob.glob(subset_path)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, subset_images), total=len(subset_images)))

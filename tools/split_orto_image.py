import glob
import os.path
import re
import concurrent.futures

import numpy as np
import torch
from PIL import Image
from torch import clamp
from tqdm import tqdm


subset_path = "C:\\Users\\lukag\\Documents\\Projects\\Faks\\MAG\\ortophoto\\fixed\\*.tif"

with_mp = True

image_normal = '../dataset/ortophoto_pretraining/train_C'
image_noise = '../dataset/ortophoto_pretraining/train_A'

tile_size = 256
resize_to = 256

"""
Split large image image into tiles.

Set tile_size var to define the size of each tile
"""
def process_image(image_path):
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

            # Generate Gaussian noise
            noise = np.random.normal(0, np.random.random_sample() * 5, tile.shape)

            # Add the noise to the image
            noisy_image = clamp(tile + noise, min=0, max=255)

            # If image is RGB
            if tile.shape[0] == 3:
                tile = tile.permute(1, 2, 0).contiguous().view(tile_size, tile_size, 3)
                noisy_image = noisy_image.permute(1, 2, 0).contiguous().view(tile_size, tile_size, 3)

            save_image(tile, image_normal, filename, idx)
            save_image(noisy_image, image_noise, filename, idx)
            idx += 1


def save_image(img, fp, fn, idx):
    img = Image.fromarray(img.numpy().astype(np.uint8), 'RGB')

    if tile_size != resize_to:
        img = img.resize((resize_to, resize_to))

    img.save(f"{fp}/{fn}-{idx}.png", "PNG")


if __name__ == '__main__':
    if not os.path.exists(image_noise):
        os.mkdir(image_noise)
    if not os.path.exists(image_normal):
        os.mkdir(image_normal)

    subset_images = glob.glob(subset_path)

    if with_mp:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_image, subset_images), total=len(subset_images)))
    else:
        for image in tqdm(subset_images, total=len(subset_images)):
            process_image(image)
import glob
import os.path

import numpy as np
import torch
from PIL import Image

in_paths = ['../mydataset/A', '../mydataset/B', '../mydataset/C']

tile_size = 280

"""
Split large image image into tiles.

Set tile_size var to define the size of each tile
"""


def main():
    for subset_path in in_paths:
        subset_images = glob.glob(subset_path + '/*.png')
        for image_path in subset_images:
            if subset_path.endswith('B'):
                # For mask images
                image = Image.open(image_path).convert('L')
            else:
                # For normal and shadowless images
                image = Image.open(image_path).convert('RGB')
            image_tensor = torch.tensor(np.array(image))
            tiles = image_tensor.unfold(0, tile_size, tile_size).unfold(1, tile_size, tile_size)

            idx = 1
            for i in range(tiles.shape[0]):
                for j in range(tiles.shape[1]):
                    tile = tiles[i, j]
                    if subset_path.endswith('B'):
                        tile_image = Image.fromarray(tile.numpy().astype(np.uint8), 'L')
                    else:
                        reshaped_image = tile.permute(1, 2, 0).contiguous().view(tile_size, tile_size, 3)
                        tile_image = Image.fromarray(reshaped_image.numpy().astype(np.uint8), 'RGB')
                    file_name = os.path.basename(image_path).split('.')[0]
                    folder_path = ' '.join(image_path.split('\\')[:-1])
                    tile_image.save(folder_path + f"/{file_name}-{idx}.jpg")
                    idx += 1


if __name__ == '__main__':
    main()

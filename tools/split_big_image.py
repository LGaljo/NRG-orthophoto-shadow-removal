import numpy as np
import torch
from PIL import Image

in_paths = ['./st-cgan/dataset/test/bledA', './st-cgan/dataset/test/bledB', './st-cgan/dataset/test/bledC']

tile_size = int(5120 / 16)


"""
Split large image image into tiles.

Set tile_size var to define the size of each tile
"""

def main():
    for image_path in in_paths:
        if image_path.endswith('B'):
            # For mask images
            image = Image.open(image_path + "/bled.jpg").convert('L')
        else:
            # For normal and shadowless images
            image = Image.open(image_path + "/bled.jpg").convert('RGB')
        image_tensor = torch.tensor(np.array(image))
        tiles = image_tensor.unfold(0, tile_size, tile_size).unfold(1, tile_size, tile_size)

        idx = 1
        for i in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                tile = tiles[i, j]
                if image_path.endswith('B'):
                    tile_image = Image.fromarray(tile.numpy().astype(np.uint8), 'L')
                else:
                    reshaped_image = tile.permute(1, 2, 0).contiguous().view(tile_size, tile_size, 3)
                    tile_image = Image.fromarray(reshaped_image.numpy().astype(np.uint8), 'RGB')
                tile_image.save(image_path + f"/bled-{idx}.jpg")
                idx += 1


if __name__ == '__main__':
    main()

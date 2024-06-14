import numpy as np
import torch
from PIL import Image

from st_cgan.main import ST_CGAN


def remove_shadows_with_mask(st_cgan):
    image = Image.open("../img/merged/DOF_D96TM_2018_2021_81533_73159_16_2024-01-22_174339.jpg")
    masks = Image.open("../img/merged/DOF_D96TM_2018_2021_81533_73159_16_2024-01-22_174339 mask.jpg").convert('L')
    tiles = (torch.tensor(np.array(image))
             .unfold(0, 256, 256).unfold(1, 256, 256))
    mask_tiles = (torch.tensor(np.array(masks))
                  .unfold(0, 256, 256).unfold(1, 256, 256))
    # merged_image = torch.zeros((rows * 256, cols * 256, image_tensor.shape[2]), dtype=torch.uint8)
    merged_image2 = Image.new("RGB", (tiles.shape[0] * 256, tiles.shape[1] * 256))

    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            start_row = i * 256
            start_col = j * 256
            tile = tiles[i, j]
            mask = mask_tiles[i, j]
            reshaped_image = tile.permute(1, 2, 0).contiguous().view(256, 256, 3)
            tile_image = Image.fromarray(reshaped_image.numpy().astype(np.uint8), 'RGB')
            mask_image = Image.fromarray(mask.numpy().astype(np.uint8), 'L')
            tile_image, _ = st_cgan.convert_image_mask(tile_image, mask_image)

            # Calculate the position of the tile in the merged image
            paste_position = (start_col, start_row)

            # Paste the tile into the merged image
            merged_image2.paste(tile_image, paste_position)

            # merged_image[start_row:start_row + 256, start_col:start_col + 256, :] = torch.tensor(np.array(tile_image))

    # merged_image_pil = Image.fromarray(merged_image.numpy())
    out_path = "../img/merged/DOF_D96TM_2018_2021_81533_73159_16_2024-01-22_174339 shadowless.jpg"
    merged_image2.save(out_path)

    print(f"Image saved to {out_path}")


def remove_shadows(st_cgan):

    image = Image.open("../13-2.png")
    tiles = (torch.tensor(np.array(image))
             .unfold(0, 256, 256).unfold(1, 256, 256))
    # merged_image = torch.zeros((rows * 256, cols * 256, image_tensor.shape[2]), dtype=torch.uint8)
    merged_image2 = Image.new("RGB", (tiles.shape[0] * 256, tiles.shape[1] * 256))

    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            start_row = i * 256
            start_col = j * 256
            tile = tiles[i, j]
            reshaped_image = tile.permute(1, 2, 0).contiguous().view(256, 256, 3)
            tile_image = Image.fromarray(reshaped_image.numpy().astype(np.uint8), 'RGB')
            tile_image, _ = st_cgan.convert_image(tile_image)

            # Calculate the position of the tile in the merged image
            paste_position = (start_col, start_row)

            # Paste the tile into the merged image
            merged_image2.paste(tile_image, paste_position)

            # merged_image[start_row:start_row + 256, start_col:start_col + 256, :] = torch.tensor(np.array(tile_image))

    # merged_image_pil = Image.fromarray(merged_image.numpy())
    out_path = "../13-2 shadowless.jpg"
    merged_image2.save(out_path)

    print(f"Image saved to {out_path}")


if __name__ == "__main__":
    # st_cgan = ST_CGAN("../st_cgan/model/istd_full/ST-CGAN_G1_2000.pth", "../st_cgan/model/istd_full/ST-CGAN_G2_2000.pth")
    st_cgan = ST_CGAN("../st_cgan/model/ST-CGAN_G1_2400.pth", "../st_cgan/model/ST-CGAN_G2_2400.pth")
    # remove_shadows(st_cgan)
    # remove_shadows_with_mask(st_cgan)

    image = Image.open("../shadowless-4-9.jpg")
    convert, _ = st_cgan.convert_image(image)
    convert.save("../shadowless-4-9 shadowless.jpg")


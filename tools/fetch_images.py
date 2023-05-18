import os

import requests
from PIL import Image

"""
Fetch images from ARSO GIS

To download the ortho-photo image define parameters:
    lid - layer id
    r - row
    c - column
    lod - level of details
    f - format of images (jpg, tiff)

The script takes the location of center tile and lod parameter.
The default size is 20x20 tiles.
"""


api_url = "http://gistiles1.arso.gov.si/nukleus_tiles2/Gis/NukleusTiles/v50/AgccTile.ashx"
params = {
    "r": 1402,
    "c": 1334,
    "lod": 16,
    "lid": "lay_ao_dof_2019",
    "f": "jpg"
}

tile_size = (20, 20)


def get_map_tiles(center_r, center_c, lod):
    params['lod'] = lod
    start_r = center_r - int(tile_size[0] / 2)
    end_r = center_r + int(tile_size[0] / 2)
    start_c = center_c - int(tile_size[1] / 2)
    end_c = center_c + int(tile_size[1] / 2)
    for r in range(start_r, end_r):
        for c in range(start_c, end_c):
            params['r'] = r
            params['c'] = c
            get_map_tile()


def get_map_tile():
    imfn = './img/' + "%s_%s_%s_%s.%s" % (
        params['lid'], str(params['r']), str(params['c']), str(params['lod']), params['f'])

    if not os.path.exists('../img'):
        os.mkdir('../img')

    if os.path.exists(imfn):
        print("File already exists!")
        return
    else:
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            # Open a new file in binary mode and write the image data to it
            with open(imfn, "wb") as f:
                f.write(response.content)
            print("Image saved successfully!")
        else:
            print("Error saving image:", response.status_code)


def merge_image_tiles(center_r, center_c, ts=tile_size):
    # Calculate the number of tiles needed to cover the image
    start_r = int(center_r - ts[0] / 2)
    end_r = int(center_r + ts[0] / 2)
    start_c = int(center_c - ts[1] / 2)
    end_c = int(center_c + ts[1] / 2)

    # Create a new blank image with the same size as the original image
    merged_image = Image.new("RGB", (tile_size[0] * 256, tile_size[1] * 256))

    # Loop through the tiles and paste them into the merged image
    for x in range(start_r, end_r):
        for y in range(start_c, end_c):
            # Open the tile image
            tile_image = Image.open('./img/' + "%s_%s_%s_%s.%s" % (
                params['lid'], str(x), str(y), str(params['lod']), params['f']))

            # Calculate the position of the tile in the merged image
            position = (y - center_c + tile_size[0] / 2), (x - center_r + tile_size[1] / 2)
            paste_position = (int(position[0] * 256), int(position[1] * 256))

            # Paste the tile into the merged image
            merged_image.paste(tile_image, paste_position)

    path = f"{params['lid']}_{center_r}_{center_c}_{params['lod']}.jpg"
    merged_image.save(path)
    return path, merged_image


if __name__ == '__main__':
    get_map_tiles(2000, 1500, 16)
    merge_image_tiles(2000, 1500)

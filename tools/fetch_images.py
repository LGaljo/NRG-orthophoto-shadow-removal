import os

import argparse as argparse
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

parser = argparse.ArgumentParser(description="Fetch images from Atlas Okolja",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gcid', default="lay_AO_DOF_2019", help="GCID parameter")
parser.add_argument('--lid', default="lay_ao_dof_2019", help="Layer id param")
parser.add_argument('--lod', default=16, type=int, help="Level of details, number between [0, 16]")
parser.add_argument('-f', default="jpg", help="Format of images (jpg or tiff)")
parser.add_argument('-r', required=True, help="Location of G1 Generator model")
parser.add_argument('-c', required=True, help="Location of G2 Generator model")
parser.add_argument('-w', '--width', default=20, type=int, help="The number of tiles per axis")
parser.add_argument('-o', '--output', default="../img", type=str, help="Output path")


class AtlasOkoljaFetcher:
    def __init__(self, args):
        self.lid = args['lid']
        self.lod = str(args['lod'])
        self.gcid = str(args['gcid'])
        self.size = (int(args['width']), int(args['width']))
        self.r = int(args['r'])
        self.c = int(args['c'])
        self.f = str(args['f'])
        self.output = str(args['output'])
        self.s_r = int(self.r) - int(self.size[0] / 2)
        self.e_r = int(self.r) + int(self.size[0] / 2)
        self.s_c = int(self.c) - int(self.size[1] / 2)
        self.e_c = int(self.c) + int(self.size[1] / 2)

        if not os.path.exists(self.output):
            os.mkdir(self.output)
            os.mkdir(os.path.join(self.output, "merged"))

    def fetch(self):
        for r in range(self.s_r, self.e_r):
            for c in range(self.s_c, self.e_c):
                self.get_tile(r, c)

    def get_tile(self, r, c):
        tile_path = os.path.join(self.output, "%s_%s_%s_%s.%s" % (self.lid, str(r), str(c), self.lod, self.f))
        params = {
            "r": str(r),
            "c": str(c),
            "lod": str(self.lod),
            "lid": str(self.lid),
            "f": str(self.f),
            "gcid": str(self.gcid),
        }
        if os.path.exists(tile_path):
            print(f"Image {os.path.basename(tile_path)} already exists!")
            return
        else:
            response = requests.get(api_url, params=params)
            if response.status_code == 200:
                with open(tile_path, "wb") as f:
                    f.write(response.content)
                print(f"Image {os.path.basename(tile_path)} saved successfully!")
            else:
                print(f"Error saving image {os.path.basename(tile_path)}: {response.status_code}")

    def merge(self):
        # Check that all images exist
        for r in range(self.s_r, self.e_r):
            for c in range(self.s_c, self.e_c):
                tile_path = os.path.join(self.output, "%s_%s_%s_%s.%s" % (self.lid, str(r), str(c), self.lod, self.f))
                if not os.path.exists(tile_path):
                    print(f"The tile r{r} c{c} does not exist")
                    exit(1)

        # Create a new blank image with the same size as the original image
        merged_image = Image.new("RGB", (self.size[0] * 256, self.size[1] * 256))

        # Loop through the tiles and paste them into the merged image
        for r in range(self.s_r, self.e_r):
            for c in range(self.s_c, self.e_c):
                tile_path = os.path.join(self.output, "%s_%s_%s_%s.%s" % (self.lid, str(r), str(c), self.lod, self.f))
                # Open the tile image
                tile_image = Image.open(tile_path)

                # Calculate the position of the tile in the merged image
                position = (c - self.c + self.size[0] / 2), (r - self.r + self.size[1] / 2)
                paste_position = (int(position[0] * 256), int(position[1] * 256))

                # Paste the tile into the merged image
                merged_image.paste(tile_image, paste_position)

        path = os.path.join(self.output, "merged", f"{self.lid}_{self.r}_{self.c}_{self.lod}.jpg")
        merged_image.save(path)
        return path, merged_image


if __name__ == '__main__':
    args = parser.parse_args()

    fetcher = AtlasOkoljaFetcher(args.__dict__)
    fetcher.fetch()
    fetcher.merge()

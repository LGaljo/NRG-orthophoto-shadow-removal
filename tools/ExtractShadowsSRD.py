import glob
import os

from tools.ExtractShadows import extract_shadow_mask_in_lab

root_path = "../SRD/SRD_Train/Train/"
image_path = "../SRD/SRD_Train/Train/shadow_free/*.jpg"
image_path_shadow = "../SRD/SRD_Train/Train/shadow/*.jpg"
image_path_mask = "../SRD/SRD_Train/Train/shadow_mask"

"""
Split large image image into tiles.

Set tile_size var to define the size of each tile
"""


def main():
    subset_images_noshadow = glob.glob(image_path)
    for ip in subset_images_noshadow:
        # extract_shadow_mask(image_path, image_path.replace('Hard', 'None'), image_path.replace('Hard', 'Mask'))
        extract_shadow_mask_in_lab(
            ip.replace('_no_shadow', '').replace('_free', ''),
            ip,
            ip.replace('free', 'mask').replace('_no_shadow', '')
        )
        os.rename(ip, ip.replace('_no_shadow', ''))


if __name__ == '__main__':
    main()

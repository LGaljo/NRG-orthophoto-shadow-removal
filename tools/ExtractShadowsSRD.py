import glob

from tools.ExtractShadows import extract_shadow_mask_in_lab

root_path = "../st_cgan/trained_models/st-cgan/srd_results/dataset"
image_path_shadow = root_path + "/test/train_A/*.jpg"
image_path_mask = root_path + "/test/train_B"
image_path = root_path + "/test/train_C/*.jpg"

"""
Split large image image into tiles.

Set tile_size var to define the size of each tile
"""


def main():
    subset_images_noshadow = glob.glob(image_path)
    for ip in subset_images_noshadow:
        extract_shadow_mask_in_lab(
            ip.replace('train_C', 'train_A'),
            ip,
            ip.replace('train_C', 'train_B')
        )


if __name__ == '__main__':
    main()

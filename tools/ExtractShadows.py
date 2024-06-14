import glob
import concurrent.futures
from PIL import Image, ImageChops, ImageEnhance, ImageCms
from tqdm import tqdm

# subset_path = "..\\unity_dataset\\big_trees_datasets\\full_size\\*.png"
subset_path = "C:\\Users\\lukag\\AppData\\LocalLow\\Magistrska naloge - Luka Galjot\\Mag Generate Shadows\\*.png"

"""
Split large image image into tiles.

Set tile_size var to define the size of each tile
"""


def rgb2labProfile():
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    return ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")


def extract_shadow_mask_in_lab(image_path, image_path_ns, image_path_adj):
    image = Image.open(image_path).convert('RGB')
    image_ns = Image.open(image_path_ns).convert('RGB')
    lab_image = ImageCms.applyTransform(image, rgb2labProfile())
    lab_image_ns = ImageCms.applyTransform(image_ns, rgb2labProfile())

    # Split into LAB color space components
    Li, ai, bi = lab_image.split()
    Lins, ains, bins = lab_image_ns.split()

    differences_image = ImageChops.difference(Li, Lins)
    # differences.save(image_path_diffs)

    # adjusted_image = differences
    adjusted_image = ImageEnhance.Brightness(differences_image).enhance(4)
    adjusted_image = ImageEnhance.Contrast(adjusted_image).enhance(1.5)

    # Save the adjusted image
    # differences.save(image_path_adj)
    adjusted_image.save(image_path_adj)


def process_image(image_path):
    image_path_ns = image_path.replace('Hard', 'None')
    image_path_adj = image_path.replace('Hard', 'Mask')
    extract_shadow_mask_in_lab(image_path, image_path_ns, image_path_adj)


if __name__ == '__main__':
    subset_images = glob.glob(subset_path)
    subset_images = list(filter(lambda s: 'Hard' in s, subset_images))

    # extract_shadow_mask_in_lab(
    #     '../unity_dataset/big_trees_datasets/full_size/DOF_D96TM_2018_2021_82327_72099_16_2024-01-22_183524_a35cf18bb6a6e239-x125-z125-Hard.png',
    #     '../unity_dataset/big_trees_datasets/full_size/DOF_D96TM_2018_2021_82327_72099_16_2024-01-22_183524_a35cf18bb6a6e239-x125-z125-None.png',
    #     '../unity_dataset/big_trees_datasets/full_size/DOF_D96TM_2018_2021_82327_72099_16_2024-01-22_183524_a35cf18bb6a6e239-x125-z125-Mask.png')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, subset_images), total=len(subset_images)))

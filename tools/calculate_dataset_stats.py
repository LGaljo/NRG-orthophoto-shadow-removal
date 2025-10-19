import numpy as np
from PIL import Image
from imutils import paths

import config
from torchvision import transforms
from tqdm import tqdm
import torch


def calculate_stats(image_paths):
    """
    Calculate mean and standard deviation of the training dataset
    """
    print("[INFO] Calculating dataset mean and standard deviation...")

    # Initialize variables to store sum and sum of squares
    pixel_sum = torch.zeros(3)
    pixel_sum_squared = torch.zeros(3)
    num_pixels = 0

    # Process images in batches to calculate mean and std
    for i, img_path in enumerate(tqdm(image_paths)):
        # Load image
        img = Image.open(img_path).convert('RGB')

        # Resize image to the input dimensions
        img = transforms.functional.resize(img, [config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH])

        # Convert to tensor (values between 0 and 1)
        img_tensor = transforms.functional.to_tensor(img)

        # Update sums
        pixel_sum += torch.sum(img_tensor, dim=[1, 2])
        pixel_sum_squared += torch.sum(img_tensor ** 2, dim=[1, 2])
        num_pixels += img_tensor.shape[1] * img_tensor.shape[2]

        # Process only a subset of images for efficiency if dataset is large
        # if i > 500:  # Adjust this number based on your dataset size
        #     break

    # Calculate mean and std
    mean = pixel_sum / num_pixels
    var = (pixel_sum_squared / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    print(f"[INFO] Dataset mean: {mean}, std: {std}")

    return mean, std

if __name__ == '__main__':
    shadow_image = []
    gt_image = []

    for image_dir in config.IMAGE_DATASET_PATHS:
        shadow_image.extend(sorted(list(paths.list_images(image_dir))))
    for image_dir in config.GT_DATASET_PATHS:
        gt_image.extend(sorted(list(paths.list_images(image_dir))))

    calculate_stats(shadow_image)
    calculate_stats(gt_image)
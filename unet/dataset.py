import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2 import ToTensor

from unet import config


class ImageLoaderDataset(Dataset):
    def __init__(self, train_paths, gt_paths, transforms):
        # store the image and mask filepaths, and augmentation transforms
        self.train_paths = train_paths
        self.gt_paths = gt_paths
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.train_paths)

    def __getitem__(self, idx):
        # load the image from disk, swap its channels to RGB,
        # and read the associated shadowless image from disk, swap its channels to RGB,
        image_train = Image.open(self.train_paths[idx]).convert('RGB')
        image_gt = Image.open(self.gt_paths[idx]).convert('RGB')

        image_train_tensor, image_gt_tensor, image_train, image_gt = self.transform(image_train, image_gt)

        # Save a copy of transformed images
        if config.SAVE_TRANSFORMS:
            image_train.save(os.path.basename(self.train_paths[idx]).split('.')[0] + "1.png")
            image_gt.save(os.path.basename(self.gt_paths[idx]).split('.')[0] + "2.png")

        # return a tuple of the image and its mask
        return image_train_tensor, image_gt_tensor

    def transform(self, image_train, image_gt):
        if 'Resize' in self.transforms:
            image_train = F.resize(image_train, size=[config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH])
            image_gt = F.resize(image_gt, size=[config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH])

        if 'RandomResizedCrop' in self.transforms:
            # Get the image dimensions (assumes image is a tensor with shape (C, H, W))
            image_height, image_width = image_train.size

            # Define a scale range and aspect ratio range similar to RandomResizedCrop
            scale_range = (0.8, 1.0)
            aspect_ratio_range = (0.75, 1.33)

            # Calculate the area of the image
            image_area = image_height * image_width

            # Generate a random scale and aspect ratio
            target_area = random.uniform(*scale_range) * image_area
            aspect_ratio = random.uniform(*aspect_ratio_range)

            # Calculate crop height and width based on the target area and aspect ratio
            crop_height = int(round((target_area / aspect_ratio) ** 0.5))
            crop_width = int(round((target_area * aspect_ratio) ** 0.5))

            # Ensure the crop size doesn't exceed the image dimensions
            crop_height = min(crop_height, image_height)
            crop_width = min(crop_width, image_width)

            # Randomly select the top and left coordinates for cropping
            top = random.randint(0, image_height - crop_height)
            left = random.randint(0, image_width - crop_width)

            final_size = [config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH]

            image_train = F.resized_crop(image_train, top, left, crop_height, crop_width, size=final_size)
            image_gt = F.resized_crop(image_gt, top, left, crop_height, crop_width, size=final_size)

        if 'ColorJitter' in self.transforms:
            # Random jitter parameters (adjust values within the specified range)
            brightness_factor = random.uniform(0.8, 1.2)  # brightness=0.2 means 1±0.2
            contrast_factor = random.uniform(0.8, 1.2)  # contrast=0.2 means 1±0.2
            saturation_factor = random.uniform(0.8, 1.2)  # saturation=0.2 means 1±0.2
            hue_factor = random.uniform(-0.1, 0.1)  # hue=0.1 means ±0.1

            # Apply jitter transformations to image_train
            image_train = F.adjust_brightness(image_train, brightness_factor)
            image_train = F.adjust_contrast(image_train, contrast_factor)
            image_train = F.adjust_saturation(image_train, saturation_factor)
            image_train = F.adjust_hue(image_train, hue_factor)

            # Apply the same jitter transformations to image_gt
            image_gt = F.adjust_brightness(image_gt, brightness_factor)
            image_gt = F.adjust_contrast(image_gt, contrast_factor)
            image_gt = F.adjust_saturation(image_gt, saturation_factor)
            image_gt = F.adjust_hue(image_gt, hue_factor)

        if 'GaussianNoise' in self.transforms:
            noise = np.random.normal(0, 5, image_train.shape)
            image_train = np.clip(image_train + noise, 0, 255).astype(np.uint8)
            image_gt = np.clip(image_gt + noise, 0, 255).astype(np.uint8)

        if 'RandomHorizontalFlip' in self.transforms:
            if random.random() > 0.5:
                image_train = F.hflip(image_train)
                image_gt = F.hflip(image_gt)

        if 'RandomVerticalFlip' in self.transforms:
            if random.random() > 0.5:
                image_train = F.vflip(image_train)
                image_gt = F.vflip(image_gt)

        if 'RandomRotation' in self.transforms:
            angle = random.choice([0, 90, 180, 270])
            image_train = F.rotate(image_train, angle=angle)
            image_gt = F.rotate(image_gt, angle=angle)

        to_tensor = ToTensor()

        return to_tensor(image_train), to_tensor(image_gt), image_train, image_gt

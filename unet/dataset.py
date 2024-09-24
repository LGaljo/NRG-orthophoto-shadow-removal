import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


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

        # apply the transformations to both images
        state = torch.get_rng_state()
        image_train = self.transforms(image_train)
        torch.set_rng_state(state)
        image_gt = self.transforms(image_gt)

        # Save a copy of transformed images
        # trimg = (image_train * 255).permute(1, 2, 0).contiguous().view(256, 256, 3)
        # trimg = Image.fromarray(trimg.numpy().astype(np.uint8), 'RGB')
        # trimg.save(os.path.basename(self.train_paths[idx]).split('.')[0] + "1.png")
        #
        # gtimg = (image_gt * 255).permute(1, 2, 0).contiguous().view(256, 256, 3)
        # gtimg = Image.fromarray(gtimg.numpy().astype(np.uint8), 'RGB')
        # gtimg.save(os.path.basename(self.gt_paths[idx]).split('.')[0] + "2.png")

        # return a tuple of the image and its mask
        return image_train, image_gt

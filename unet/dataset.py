from torch.utils.data import Dataset
from PIL import Image


class ImageLoaderDataset(Dataset):
    def __init__(self, shadow_paths, shadowless_paths, transforms):
        # store the image and mask filepaths, and augmentation transforms
        self.shadow_paths = shadow_paths
        self.shadowless_paths = shadowless_paths
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.shadow_paths)

    def __getitem__(self, idx):
        # load the image from disk, swap its channels to RGB,
        # and read the associated shadowless image from disk, swap its channels to RGB,
        shadow_image = Image.open(self.shadow_paths[idx]).convert('RGB')
        shadowless_image = Image.open(self.shadowless_paths[idx]).convert('RGB')

        # apply the transformations to both images
        shadow_image = self.transforms(shadow_image)
        shadowless_image = self.transforms(shadowless_image)

        # return a tuple of the image and its mask
        return shadow_image, shadowless_image

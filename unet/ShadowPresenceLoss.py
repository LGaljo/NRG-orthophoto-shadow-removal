import numpy as np
import torch
from skimage.filters import threshold_multiotsu
from torch import Tensor
from torch.nn import Module
from torchvision.transforms.v2.functional import rgb_to_grayscale


class SPLoss(Module):
    def __init__(self, alpha=0.5):
        super(SPLoss, self).__init__()
        self.alpha = alpha

    def forward(self, images, y):
        if not isinstance(images, Tensor):
            raise TypeError("Input should be a PyTorch Tensor")

        gray_images = rgb_to_grayscale(images)
        rgb_images = np.array(images.detach().to('cpu').numpy())
        gray_images = np.array(gray_images.detach().to('cpu').numpy())

        shadow_pix_share = []
        for idx in range(rgb_images.shape[0]):
            image = rgb_images[idx]
            gray_image = gray_images[idx].squeeze()
            shadow_pix_sum = self.multiotsu_threshold(gray_image, image)
            shadow_pix_share.append(shadow_pix_sum / (gray_image.shape[0] * gray_image.shape[1]))

        shadow_pix_share = np.array(shadow_pix_share).mean()

        return torch.tensor(shadow_pix_share, device=images.device, requires_grad=False)

    def multiotsu_threshold(self, gray_image, rgb_image):
        thresholds = threshold_multiotsu(gray_image)

        suspected_shadows = np.digitize(gray_image, bins=[thresholds[0]])

        # Convert mask to boolean
        # suspected_shadows = foreground.astype(bool)

        # Create an empty mask for true shadows
        true_shadows = np.zeros_like(gray_image, dtype=int)

        # Split the image into B, G, R channels
        # R = rgb_image[0, :, :]  # Red channel
        G = rgb_image[1, :, :]  # Green channel
        B = rgb_image[2, :, :]  # Blue channel

        Ga = 16

        # Iterate over each suspected shadow to determine if it's a true shadow or false
        for i in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1]):
                if suspected_shadows[i, j] != 1:
                    if B[i, j] + Ga < G[i, j]:
                        # It's vegetation, rule out as shadow
                        continue
                    else:
                        # It's a true shadow
                        true_shadows[i, j] = 1

        return sum(map(sum, true_shadows))
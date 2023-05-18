from st_cgan.ST_CGAN import Generator
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
from PIL.Image import Resampling

import torch
import os

torch.manual_seed(44)
# choose your device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def fix_model_state_dict(state_dict):
    """
    remove 'module.' of dataparallel
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def unnormalize(x):
    x = x.transpose(1, 3)
    # mean, std
    x = x * torch.Tensor((0.5,)) + torch.Tensor((0.5,))
    x = x.transpose(1, 3)
    return x


class ST_CGAN:
    def __init__(self, path_g1, path_g2):
        self.G1 = Generator(input_channels=3, output_channels=1)
        self.G2 = Generator(input_channels=4, output_channels=3)

        '''load'''
        print(f'Load model {path_g1} and {path_g2}')

        G1_weights = torch.load(path_g1)
        self.G1.load_state_dict(fix_model_state_dict(G1_weights))

        G2_weights = torch.load(path_g2)
        self.G2.load_state_dict(fix_model_state_dict(G2_weights))

        mean = (0.5,)
        std = (0.5,)

        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.G1.to(self.device)
        self.G2.to(self.device)

        """use GPU in parallel"""
        if self.device == 'cuda':
            self.G1 = torch.nn.DataParallel(self.G1)
            self.G2 = torch.nn.DataParallel(self.G2)
            print("parallel mode")

        print("device: {}".format(self.device))

        self.G1.eval()
        self.G2.eval()

    def convert_image(self, image):
        img = image.convert('RGB')
        width, height = img.width, img.height
        img = self.img_transform(img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            detected_shadow = self.G1(img.to(self.device))
            detected_shadow = detected_shadow.to(torch.device('cpu'))
            concat = torch.cat([img, detected_shadow], dim=1)
            shadow_removal_image = self.G2(concat.to(self.device))
            shadow_removal_image = shadow_removal_image.to(torch.device('cpu'))

            shadow_removal_image = transforms.ToPILImage(mode='RGB')(unnormalize(shadow_removal_image)[0, :, :, :])
            shadow_removal_image = shadow_removal_image.resize((width, height), Resampling.LANCZOS)
            # shadow_removal_image.save(out_path)
            return shadow_removal_image, transforms.ToPILImage(mode='L')(unnormalize(detected_shadow)[0, :, :, :])

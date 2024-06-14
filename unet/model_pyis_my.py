# import the necessary packages
from torch import nn

import config
from torch.nn import ConvTranspose2d, BatchNorm2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.BatchNorm = BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        # CONV -> BATCH NORM -> RELU -> CONV -> BATCH NORM -> RELU
        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.BatchNorm(x)
        x = self.relu(x)

        return x


class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super(Encoder, self).__init__()
        self.encBlocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        block_outputs = []

        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            p = self.pool(x)
            block_outputs.append((p, x))

        # return the list containing the intermediate outputs
        return block_outputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

    def forward(self, p, x):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            x = torch.cat([x, p], dim=3)
            x = self.dec_blocks[i](x)

        # return the final decoder output
        return x


class UNet(Module):
    def __init__(self, enc_channels=(3, 16, 32, 64), dec_channels=(64, 32, 16),
                 nb_classes=3, out_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        # initialize the regression head and store the class variables
        self.head = Conv2d(dec_channels[-1], nb_classes, 1)
        self.out_size = out_size

    def forward(self, x):
        # grab the features from the encoder
        enc_features = self.encoder(x)
        # b1 = Block(1024, 1024)(x)
        dec_features = self.decoder(enc_features)

        # pass the decoder features through the regression head to obtain the segmentation mask
        segmentation_mask = self.head(dec_features)

        # return the segmentation map
        return segmentation_mask

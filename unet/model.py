# import the necessary packages
from torch import nn

import config
from torch.nn import BatchNorm2d, Sequential, ConvTranspose2d, Conv2d, MaxPool2d, Module, ModuleList, ReLU
from torch.nn import functional as F
from torch import cat


class BasicBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, norm=False, relu=True, transpose=False):
        super(BasicBlock, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias))

        if norm:
            layers.append(BatchNorm2d(out_channels))
        if relu:
            layers.append(ReLU(inplace=True))

        self.main = Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.main = Sequential(
            BasicBlock(in_channels, out_channels, kernel_size=3, stride=1, relu=True),
            BasicBlock(in_channels, out_channels, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x):
        return self.main(x) + x


class Encoder(Module):
    def __init__(self, out_channel, num_res):
        super(Encoder, self).__init__()

        layers = [Block(out_channel, out_channel) for _ in range(num_res)]
        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(Module):
    def __init__(self, channel, num_res=8):
        super(Decoder, self).__init__()

        layers = [Block(channel, channel) for _ in range(num_res)]
        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicBlock(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicBlock(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicBlock(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicBlock(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicBlock(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicBlock(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicBlock(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicBlock(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class UNet(Module):
    def __init__(self, num_res=20):
        super().__init__()

        base_channel = 32
        self.Encoder = nn.ModuleList([
            Encoder(base_channel, num_res),
            Encoder(base_channel*2, num_res),
            Encoder(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicBlock(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicBlock(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicBlock(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicBlock(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicBlock(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicBlock(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            Decoder(base_channel * 4, num_res),
            Decoder(base_channel * 2, num_res),
            Decoder(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicBlock(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicBlock(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList([
            BasicBlock(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
            BasicBlock(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        # x_2 = F.interpolate(x, scale_factor=0.5)
        # x_4 = F.interpolate(x_2, scale_factor=0.5)
        # z2 = self.SCM2(x_2)
        # z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        # z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        # z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        # z12 = F.interpolate(res1, scale_factor=0.5)
        # z21 = F.interpolate(res2, scale_factor=2)
        # z42 = F.interpolate(z, scale_factor=2)
        # z41 = F.interpolate(z42, scale_factor=2)

        # res2 = self.AFFs[1](z12, res2, z42)
        # res1 = self.AFFs[0](res1, z21, z41)
        #
        # res2 = self.drop2(res2)
        # res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        # z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        # outputs.append(z_+x_4)

        # z = cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        # z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        # outputs.append(z_+x_2)

        # z = cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)

        return z+x
        # outputs.append(z+x)
        #
        # return outputs[2]

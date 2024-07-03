import config
from torch.nn import ConvTranspose2d, BatchNorm2d, Dropout, Conv2d, MaxPool2d, Module, ModuleList, ReLU, Sigmoid, \
    Upsample, Sequential
from torch import cat, sum, mul, add


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        if config.BATCH_NORM:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if config.BATCH_NORM:
            x = self.bn(x)
        x = self.relu(x)

        return x


class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super(Encoder, self).__init__()
        self.encBlocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        block_outputs = []

        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            block_outputs.append(x)
            x = self.maxpool(x)

        # return the list containing the intermediate outputs
        return x, block_outputs


class Decoder(Module):
    def __init__(self, ch=(64, 32, 16)):
        super().__init__()
        self.channels = ch
        self.upconvs = ModuleList([ConvTranspose2d(ch[i], ch[i + 1], kernel_size=2, stride=2, padding=0) for i in range(len(ch) - 1)])
        self.dec_blocks = ModuleList([Block(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])
        self.attentions = ModuleList([AttentionBlock(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])
        self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, link_features):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            att = self.attentions[i](x, link_features[len(self.channels) - i - 2])
            x = cat([x, att], dim=1)
            x = self.dec_blocks[i](x)

        # return the final decoder output
        return x


class Bridge(Module):
    def __init__(self, in_channels):
        super(Bridge, self).__init__()
        out_channels = in_channels * 2
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        if config.BATCH_NORM:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        if config.BATCH_NORM:
            x = self.bn(x)
        x = self.relu(x)
        return x


class AttentionBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1, padding=1, stride=1),
            BatchNorm2d(out_channels)
        )
        self.W_x = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1, padding=1, stride=1),
            BatchNorm2d(out_channels)
        )
        self.psi = Sequential(
            Conv2d(out_channels, 1, kernel_size=1, padding=1, stride=1),
            BatchNorm2d(1),
            Sigmoid()
        )
        self.relu = ReLU(inplace=True)

    def forward(self, dec, enc_link):
        g1 = self.W_g(dec)
        x1 = self.W_x(enc_link)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return enc_link * psi


class UNet(Module):
    def __init__(self, enc_channels=(3, 64, 128, 256, 512), dec_channels=(1024, 512, 256, 128, 64),
                 nb_classes=config.NUM_CHANNELS, out_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        self.bridge = Bridge(enc_channels[-1])

        # initialize the regression head and store the class variables
        self.head = Conv2d(dec_channels[-1], nb_classes, 1)
        self.out_size = out_size

    def forward(self, x):
        # grab the features from the encoder
        p, link_features = self.encoder(x)
        b = self.bridge(p)
        dec_features = self.decoder(b, link_features)

        # pass the decoder features through the regression head to obtain the segmentation mask
        out = self.head(dec_features)

        # return the segmentation map
        return out

from unet import config
from torch.nn import ConvTranspose2d, BatchNorm2d, Dropout, Conv2d, MaxPool2d, Module, ReLU, Sigmoid, Sequential
from torch import cat, mul, add


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Dropout(0.25),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Dropout(0.25),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.encBlock = Block(in_channels, out_channels)
        self.max_pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.encBlock(x)
        p = self.max_pool(x)

        return x, p


class Decoder(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconvs = ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.dec_block = Block(in_channels, out_channels)
        self.attentions = AttentionBlock(in_channels, out_channels)

    def forward(self, x, enc_link):
        x = self.upconvs(x)
        att = self.attentions(x, enc_link)
        x = cat([x, att], dim=1)
        x = self.dec_block(x)

        return x


class Bottleneck(Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.bottleneck = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bottleneck(x)


class AttentionBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        in_channels = in_channels // 2
        out_channels = out_channels // 2
        self.W_g = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1),
            BatchNorm2d(out_channels)
        )
        self.W_x = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1),
            BatchNorm2d(out_channels)
        )
        self.psi = Sequential(
            Conv2d(out_channels, 1, kernel_size=1, padding=0, stride=1),
            BatchNorm2d(1),
            Sigmoid()
        )
        self.relu = ReLU(inplace=True)

    def forward(self, dec, enc_link):
        g = self.W_g(dec)
        x = self.W_x(enc_link)
        psi = add(g, x)
        psi = self.relu(psi)
        psi = self.psi(psi)
        psi = mul(enc_link, psi)
        return psi


class UNet(Module):
    def __init__(self, nb_classes=config.NUM_CHANNELS):
        super().__init__()

        self.enc1 = Encoder(in_channels=nb_classes, out_channels=64)
        self.enc2 = Encoder(in_channels=64, out_channels=128)
        self.enc3 = Encoder(in_channels=128, out_channels=256)
        self.enc4 = Encoder(in_channels=256, out_channels=512)

        self.bottleneck = Bottleneck(in_channels=512, out_channels=1024)

        self.dec4 = Decoder(in_channels=1024, out_channels=512)
        self.dec3 = Decoder(in_channels=512, out_channels=256)
        self.dec2 = Decoder(in_channels=256, out_channels=128)
        self.dec1 = Decoder(in_channels=128, out_channels=64)

        self.head = Conv2d(64, nb_classes, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        enc1, pool1 = self.enc1(x)
        enc2, pool2 = self.enc2(pool1)
        enc3, pool3 = self.enc3(pool2)
        enc4, pool4 = self.enc4(pool3)

        b1 = self.bottleneck(pool4)

        dec4 = self.dec4(b1, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        # pass the decoder features through the regression head to obtain the segmentation mask
        out = self.head(dec1)
        out = self.sigmoid(out)

        # return the segmentation map
        return out

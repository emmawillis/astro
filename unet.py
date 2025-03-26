import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Two 3×3 convolutions + BatchNorm + ReLU, used in both encoder and decoder.
    """
    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(UNetEncoder, self).__init__()

        self.enc32 = ConvBlock(in_channels, 32)
        self.enc64 = ConvBlock(32, 64)
        self.enc128 = ConvBlock(64, 128)
        self.enc256 = ConvBlock(128, 256)
        self.enc512 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        e32 = self.enc32(x)
        p32 = self.pool(e32)

        e64 = self.enc64(p32)
        p64 = self.pool(e64)

        e128 = self.enc128(p64)
        p128 = self.pool(e128)

        e256 = self.enc256(p128)
        p256 = self.pool(e256)

        e512 = self.enc512(p256)

        return e32, e64, e128, e256, e512

class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super(UNetDecoder, self).__init__()

        self.upconv256 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec256 = ConvBlock(512, 256)

        self.upconv128 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec128 = ConvBlock(256, 128)

        self.upconv64 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec64 = ConvBlock(128, 64)

        self.upconv32 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec32 = ConvBlock(64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        # NOTE: No activation like Sigmoid — to allow full range output

    def forward(self, e512, e256, e128, e64, e32):
        d256 = self.upconv256(e512)
        d256 = self.dec256(torch.cat([d256, e256], dim=1))

        d128 = self.upconv128(d256)
        d128 = self.dec128(torch.cat([d128, e128], dim=1))

        d64 = self.upconv64(d128)
        d64 = self.dec64(torch.cat([d64, e64], dim=1))

        d32 = self.upconv32(d64)
        d32 = self.dec32(torch.cat([d32, e32], dim=1))

        out = self.final_conv(d32)
        return out

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

    def forward(self, x):
        e32, e64, e128, e256, e512 = self.encoder(x)
        out = self.decoder(e512, e256, e128, e64, e32)
        return out

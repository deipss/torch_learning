import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    '''(convolution => [BN] => ReLU) * 2'''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    '''Downscaling with maxpool then double conv'''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    '''Upscaling then double conv'''

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the size
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is x1, x2 output from downsample path
        x1 = torch.cat([x1, x2], dim=1)
        return self.conv(x1)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.d1 = Down(n_channels, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)
        self.d5 = Down(512, 512)
        self.u1 = Up(1024, 256)  # input is cat of [512, 512]
        self.u2 = Up(512, 128)   # input is cat of [256, 128]
        self.u3 = Up(256, 64)
        self.u4 = Up(128, 64)
        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Downsampling path
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)

        # Upsampling path
        u1 = self.u1(d5, d4)
        u2 = self.u2(u1, d3)
        u3 = self.u3(u2, d2)
        u4 = self.u4(u3, d1)

        # Output layer
        return self.outconv(u4)

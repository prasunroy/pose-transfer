import torch
import torch.nn as nn


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)


def downconv2x(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)


def upconv2x(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)


class ResidualBlock(nn.Module):
    
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        layers = [
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels)
        ]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        y = self.layers(x) + x
        return y


class NetG(nn.Module):
    
    def __init__(self, in1_channels, in2_channels, out_channels, ngf=64):
        super(NetG, self).__init__()
        
        self.in1_conv1 = self.inconv(in1_channels, ngf)
        self.in1_down1 = self.down2x(ngf, ngf*2)
        self.in1_down2 = self.down2x(ngf*2, ngf*4)
        self.in1_down3 = self.down2x(ngf*4, ngf*8)
        self.in1_down4 = self.down2x(ngf*8, ngf*16)
        
        self.in2_conv1 = self.inconv(in2_channels, ngf)
        self.in2_down1 = self.down2x(ngf, ngf*2)
        self.in2_down2 = self.down2x(ngf*2, ngf*4)
        self.in2_down3 = self.down2x(ngf*4, ngf*8)
        self.in2_down4 = self.down2x(ngf*8, ngf*16)
        
        self.out_up1 = self.up2x(ngf*16, ngf*8)
        self.out_up2 = self.up2x(ngf*8, ngf*4)
        self.out_up3 = self.up2x(ngf*4, ngf*2)
        self.out_up4 = self.up2x(ngf*2, ngf)
        self.out_conv1 = self.outconv(ngf, out_channels)
    
    def inconv(self, in_channels, out_channels):
        return nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def outconv(self, in_channels, out_channels):
        return nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            conv1x1(in_channels, out_channels),
            nn.Tanh()
        )
    
    def down2x(self, in_channels, out_channels):
        return nn.Sequential(
            downconv2x(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
    
    def up2x(self, in_channels, out_channels):
        return nn.Sequential(
            upconv2x(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
    
    def forward(self, x1, x2):
        x1_c1 = self.in1_conv1(x1)
        x1_d1 = self.in1_down1(x1_c1)
        x1_d2 = self.in1_down2(x1_d1)
        x1_d3 = self.in1_down3(x1_d2)
        x1_d4 = self.in1_down4(x1_d3)
        
        x2_c1 = self.in2_conv1(x2)
        x2_d1 = self.in2_down1(x2_c1)
        x2_d2 = self.in2_down2(x2_d1)
        x2_d3 = self.in2_down3(x2_d2)
        x2_d4 = self.in2_down4(x2_d3)
        
        y = x1_d4 * torch.sigmoid(x2_d4)
        y = self.out_up1(y)
        y = y * torch.sigmoid(x2_d3)
        y = self.out_up2(y)
        y = y * torch.sigmoid(x2_d2)
        y = self.out_up3(y)
        y = y * torch.sigmoid(x2_d1)
        y = self.out_up4(y)
        y = self.out_conv1(y)
        return y

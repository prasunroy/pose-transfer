import torch.nn as nn


class NetD(nn.Module):
    
    def __init__(self, in_channels, ndf=64, num_blocks=3):
        super(NetD, self).__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, True)
        ]
        n1 = 1
        n2 = 1
        for i in range(num_blocks):
            n1 = n2
            n2 = min(2**(i+1), 8)
            stride = min((num_blocks-i), 2)
            layers.extend([
                nn.Conv2d(n1*ndf, n2*ndf, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(n2*ndf),
                nn.LeakyReLU(0.2, True)
            ])
        layers.extend([
            nn.Conv2d(n2*ndf, 1, kernel_size=4, stride=1, padding=1, bias=True)
        ])
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

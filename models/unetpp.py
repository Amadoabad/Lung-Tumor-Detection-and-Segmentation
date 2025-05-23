import torch
import torch.nn as nn 
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same',  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same',  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.depth = len(features)

        self.down_convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        for f in features:
            self.down_convs.append(DoubleConv(in_channels, f))
            in_channels = f

        self.up_convs = nn.ModuleDict()
        for d in range(self.depth - 1):
            for i in range(self.depth - d - 1):
                self.up_convs[f'up_{i}_{d}'] = DoubleConv(features[i] + (d + 1) * features[i + 1], features[i])

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        X = [[None]*self.depth for _ in range(self.depth)]
        for i in range(self.depth):
            x = self.down_convs[i](x)
            X[i][0] = x
            if i < self.depth - 1:
                x = self.pool(x)

        for d in range(1, self.depth):
            for i in range(self.depth - d):
                ups = [X[i][0]] + [F.interpolate(X[i + 1][k], scale_factor=2, mode='bilinear', align_corners=True) for k in range(d)]
                X[i][d] = self.up_convs[f'up_{i}_{d - 1}'](torch.cat(ups, dim=1))

        return self.final(X[0][self.depth - 1])
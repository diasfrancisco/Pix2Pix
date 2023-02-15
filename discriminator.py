import torch
import torch.nn as nn


class PatchGAN(nn.Module):
    def __init__(self, features: int, in_dim: int, out_dim: int):
        super().__init__()
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        ) #128x128 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.features, self.features*2, 4, 2, 1, padding_mode="reflect"),
            nn.BatchNorm2d(self.features*2),
            nn.LeakyReLU(0.2)
        ) #64x64 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.features*2, self.features*4, 4, 2, 1, padding_mode="reflect"),
            nn.BatchNorm2d(self.features*4),
            nn.LeakyReLU(0.2)
        ) #32x32 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.features*4, self.features*8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(self.features*8),
            nn.LeakyReLU(0.2)
        ) #31x31 512
        self.final = nn.Sequential(
            nn.Conv2d(self.features*8, out_dim, 4, 1, padding_mode="reflect"),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        x0 = torch.cat([x, y], dim=1)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        return self.final(x4)
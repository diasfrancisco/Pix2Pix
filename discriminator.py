import torch.nn as nn
from generator import CNNBlock

class PatchGAN(nn.Module):
    def __init__(self, features: int, in_dim: int, out_dim: int):
        super().__init__()
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.conv1 = CNNBlock(in_channels=self.in_dim, out_channels=self.features).encoder_block(batch_norm=False) #128x128 64
        self.conv2 = CNNBlock(in_channels=self.features, out_channels=self.features*2).encoder_block(batch_norm=True) #64x64 128
        self.conv3 = CNNBlock(in_channels=self.features*2, out_channels=self.features*4).encoder_block(batch_norm=True) #32x32 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.features*4, self.features*8, 4, 1, bias=False, padding_mode="reflect"), #31x31 512
            nn.BatchNorm2d(self.features*8),
            nn.LeakyReLU()
        )
        self.final = nn.Sequential(
            nn.Conv2d(self.features*8, out_dim, 4, 1, padding_mode="reflect"),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x1 = self.conv1(x.float())
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        return self.final(x4)
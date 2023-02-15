import torch
import torch.nn as nn

import config


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, enc: bool, dropout: bool):
        super(CNNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.dropout_inst = nn.Dropout()
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if enc else nn.ConvTranspose2d(self.in_channels, self.out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.LeakyReLU(negative_slope=0.2)
            if enc else nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout_inst(x) if self.dropout else x

class UNet(nn.Module):
    def __init__(self, features: int, in_dim: int, out_dim: int):
        super().__init__()
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.e1 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.e2 = CNNBlock(in_channels=self.features, out_channels=self.features*2, dropout=False, enc=True) #128
        self.e3 = CNNBlock(in_channels=self.features*2, out_channels=self.features*4, dropout=False, enc=True) #256
        self.e4 = CNNBlock(in_channels=self.features*4, out_channels=self.features*8, dropout=False, enc=True) #512
        self.e5 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8, dropout=False, enc=True) #512
        self.e6 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8, dropout=False, enc=True) #512
        self.e7 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8, dropout=False, enc=True) #512
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )
        
        self.d1 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8, dropout=True, enc=False) #512
        self.d2 = CNNBlock(in_channels=self.features*16, out_channels=self.features*8, dropout=True, enc=False) #1024
        self.d3 = CNNBlock(in_channels=self.features*16, out_channels=self.features*8, dropout=True, enc=False) #1024
        self.d4 = CNNBlock(in_channels=self.features*16, out_channels=self.features*8, dropout=False, enc=False) #1024
        self.d5 = CNNBlock(in_channels=self.features*16, out_channels=self.features*4, dropout=False, enc=False) #512
        self.d6 = CNNBlock(in_channels=self.features*8, out_channels=self.features*2, dropout=False, enc=False) #256
        self.d7 = CNNBlock(in_channels=self.features*4, out_channels=self.features, dropout=False, enc=False) #128
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, self.out_dim, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        down1 = self.e1(x.float())
        down2 = self.e2(down1)
        down3 = self.e3(down2)
        down4 = self.e4(down3)
        down5 = self.e5(down4)
        down6 = self.e6(down5)
        down7 = self.e7(down6)
        
        x_bottleneck = self.bottleneck(down7)
        
        up1 = self.d1(x_bottleneck)
        up2 = self.d2(torch.cat((up1, down7), dim=1))
        up3 = self.d3(torch.cat((up2, down6), dim=1))
        up4 = self.d4(torch.cat((up3, down5), dim=1))
        up5 = self.d5(torch.cat((up4, down4), dim=1))
        up6 = self.d6(torch.cat((up5, down3), dim=1))
        up7 = self.d7(torch.cat((up6, down2), dim=1))
        up8 = self.final(torch.cat((up7, down1), dim=1))

        return up8
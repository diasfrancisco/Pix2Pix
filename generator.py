import torch
import torch.nn as nn

import config


class CNNBlock:
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_layers = [
            nn.Conv2d(self.in_channels, self.out_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        self.decoder_layers = [
            nn.ConvTranspose2d(self.in_channels, self.out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU()
        ]
            
    def encoder_block(self, batch_norm: bool):
        if batch_norm == True:
            self.encoder_layers.insert(
                1,
                nn.BatchNorm2d(num_features=self.out_channels)
            )
            self.encoder = nn.Sequential(*self.encoder_layers)
            return self.encoder
        else:
            self.encoder = nn.Sequential(*self.encoder_layers)
            return self.encoder
    
    def decoder_block(self, dropout: bool):
        if dropout == True:
            self.decoder_layers.insert(
                2,
                nn.Dropout2d(p=0.5)
            )
            self.decoder = nn.Sequential(*self.decoder_layers)
            return self.decoder
        else:
            self.decoder = nn.Sequential(*self.decoder_layers)
            return self.decoder

class UNet(nn.Module):
    def __init__(self, features: int, in_dim: int, out_dim: int):
        super().__init__()
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.e1 = CNNBlock(in_channels=self.in_dim, out_channels=self.features).encoder_block(batch_norm=False).to(device=config.DEVICE) #64
        self.e2 = CNNBlock(in_channels=self.features, out_channels=self.features*2).encoder_block(batch_norm=True).to(device=config.DEVICE) #128
        self.e3 = CNNBlock(in_channels=self.features*2, out_channels=self.features*4).encoder_block(batch_norm=True).to(device=config.DEVICE) #256
        self.e4 = CNNBlock(in_channels=self.features*4, out_channels=self.features*8).encoder_block(batch_norm=True).to(device=config.DEVICE) #512
        self.e5 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).encoder_block(batch_norm=True).to(device=config.DEVICE) #512
        self.e6 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).encoder_block(batch_norm=True).to(device=config.DEVICE) #512
        self.e7 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).encoder_block(batch_norm=True).to(device=config.DEVICE) #512
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )
        
        self.d1 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).decoder_block(dropout=True).to(device=config.DEVICE) #512
        self.d2 = CNNBlock(in_channels=self.features*16, out_channels=self.features*8).decoder_block(dropout=True).to(device=config.DEVICE) #1024
        self.d3 = CNNBlock(in_channels=self.features*16, out_channels=self.features*8).decoder_block(dropout=True).to(device=config.DEVICE) #1024
        self.d4 = CNNBlock(in_channels=self.features*16, out_channels=self.features*8).decoder_block(dropout=False).to(device=config.DEVICE) #1024
        self.d5 = CNNBlock(in_channels=self.features*16, out_channels=self.features*4).decoder_block(dropout=False).to(device=config.DEVICE) #512
        self.d6 = CNNBlock(in_channels=self.features*8, out_channels=self.features*2).decoder_block(dropout=False).to(device=config.DEVICE) #256
        self.d7 = CNNBlock(in_channels=self.features*4, out_channels=self.features).decoder_block(dropout=False).to(device=config.DEVICE) #128
        
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
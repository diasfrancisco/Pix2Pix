import torch
import torch.nn as nn

import config


class CNNBlock:
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_layers = [
            nn.Conv2d(self.in_channels, self.out_channels, 4, 2, 1),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        self.decoder_layers = [
            nn.ConvTranspose2d(self.in_channels, self.out_channels, 4, 2, 1),
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
        
    def forward(self, x):
        self.e1 = CNNBlock(in_channels=self.in_dim, out_channels=self.features).encoder_block(batch_norm=False).to(device=config.DEVICE) #64
        self.e2 = CNNBlock(in_channels=self.features, out_channels=self.features*2).encoder_block(batch_norm=True).to(device=config.DEVICE) #128
        self.e3 = CNNBlock(in_channels=self.features*2, out_channels=self.features*4).encoder_block(batch_norm=True).to(device=config.DEVICE) #256
        self.e4 = CNNBlock(in_channels=self.features*4, out_channels=self.features*8).encoder_block(batch_norm=True).to(device=config.DEVICE) #512
        self.e5 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).encoder_block(batch_norm=True).to(device=config.DEVICE) #512
        self.e6 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).encoder_block(batch_norm=True).to(device=config.DEVICE) #512
        self.e7 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).encoder_block(batch_norm=True).to(device=config.DEVICE) #512
        self.e8 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).encoder_block(batch_norm=True).to(device=config.DEVICE) #512
        
        self.d1 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).decoder_block(dropout=True).to(device=config.DEVICE) #512
        self.d2 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).decoder_block(dropout=True).to(device=config.DEVICE) #1024
        self.d3 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).decoder_block(dropout=True).to(device=config.DEVICE) #1024
        self.d4 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).decoder_block(dropout=False).to(device=config.DEVICE) #1024
        self.d5 = CNNBlock(in_channels=self.features*8, out_channels=self.features*8).decoder_block(dropout=False).to(device=config.DEVICE) #1024
        self.d6 = CNNBlock(in_channels=self.features*8, out_channels=self.features*4).decoder_block(dropout=False).to(device=config.DEVICE) #512
        self.d7 = CNNBlock(in_channels=self.features*4, out_channels=self.features*2).decoder_block(dropout=False).to(device=config.DEVICE) #256
        self.d8 = CNNBlock(in_channels=self.features*2, out_channels=self.out_dim).decoder_block(dropout=False).to(device=config.DEVICE) #128
        
        x1 = self.e1(x.float())
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        x6 = self.e6(x5)
        x7 = self.e7(x6)
        x8 = self.e8(x7)
        
        x9 = self.d1(x8)
        print(x9.shape, x7.shape)
        x10 = self.d2(torch.cat((x9, x7), dim=1))
        x11 = self.d3(torch.cat((x10, x6), dim=0))
        x12 = self.d4(torch.cat((x11, x5), dim=0))
        x13 = self.d5(torch.cat((x12, x4), dim=0))
        x14 = self.d6(torch.cat((x13, x3), dim=0))
        x15 = self.d7(torch.cat((x14, x2), dim=0))
        x16 = self.d8(torch.cat((x15, x1), dim=0))
        print(x16.shape)
        
        y_gen = nn.Tanh(x16)
        
        return y_gen
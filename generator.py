import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, dropout):
        super().__init__()
        encoder_layers = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        decoder_layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(),
            nn.ReLU()
        ]
        
        if batch_norm == True:
            encoder_layers.insert(
                1,
                nn.BatchNorm2d()
            )
            self.encoder = nn.Sequential(*encoder_layers)
        else:
            self.encoder = nn.Sequential(*encoder_layers)
        
        if dropout == True:
            decoder_layers.insert(
                2,
                nn.Dropout2d(p=0.5)
            )
            self.decoder = nn.Sequential(*decoder_layers)
        else:
            self.decoder = nn.Sequential(*decoder_layers)

class Generator(nn.Module):  
    def __init__(self, features, in_dim, out_dim):
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        
    def forward(self):
        self.e1 = CNNBlock(in_channels=self.in_dim, out_channels=self.features, batch_norm=False) #64
        self.e2 = CNNBlock(in_channels=self.e1, out_channels=self.features*2, batch_norm=True) #128
        self.e3 = CNNBlock(in_channels=self.e2, out_channels=self.features*4, batch_norm=True) #256
        self.e4 = CNNBlock(in_channels=self.e3, out_channels=self.features*8, batch_norm=True) #512
        self.e5 = CNNBlock(in_channels=self.e4, out_channels=self.features*8, batch_norm=True) #512
        self.e6 = CNNBlock(in_channels=self.e5, out_channels=self.features*8, batch_norm=True) #512
        self.e7 = CNNBlock(in_channels=self.e6, out_channels=self.features*8, batch_norm=True) #512
        self.e8 = CNNBlock(in_channels=self.e7, out_channels=self.features*8, batch_norm=True) #512
        
        self.d1 = CNNBlock(in_channels=self.e8, out_channels=self.features*8, dropout=True) #512
        self.d2 = CNNBlock(in_channels=torch.cat(self.d1, self.e7, 1), out_channels=self.features*8, dropout=True) #1024
        self.d3 = CNNBlock(in_channels=torch.cat(self.d2, self.e6, 1), out_channels=self.features*8, dropout=True) #1024
        self.d4 = CNNBlock(in_channels=torch.cat(self.d3, self.e5, 1), out_channels=self.features*8, dropout=False) #1024
        self.d5 = CNNBlock(in_channels=torch.cat(self.d4, self.e4, 1), out_channels=self.features*8, dropout=False) #1024
        self.d6 = CNNBlock(in_channels=torch.cat(self.d5, self.e3, 1), out_channels=self.features*4, dropout=False) #512
        self.d7 = CNNBlock(in_channels=torch.cat(self.d6, self.e2, 1), out_channels=self.features*2, dropout=False) #256
        self.d8 = CNNBlock(in_channels=torch.cat(self.d7, self.e1, 1), out_channels=self.out_dim, dropout=False) #128
        
        self.final = nn.Tanh(self.d8)
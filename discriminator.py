import torch.nn as nn
from generator import CNNBlock

class PatchGAN():
    def __init__(self, features: int, in_dim: int, out_dim: int):
        super().__init__()
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.conv1 = CNNBlock(in_channels=self.in_dim, out_channels=self.features, batch_norm=False)
        self.conv2 = CNNBlock(in_channels=self.conv1, out_channels=self.features, batch_norm=True)
        self.conv3 = CNNBlock(in_channels=self.in_dim, out_channels=self.features, batch_norm=True)
        self.conv4 = CNNBlock(in_channels=self.in_dim, out_channels=self.features, batch_norm=True)
        
    def forward(self, y_gen):
        pass
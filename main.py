import torch
from generator import UNet
from dataset import Pix2PixDataset
import matplotlib.pyplot as plt

import config


def run():
    dataset = Pix2PixDataset(root_dir=config.ROOT_DIR, transform=True).__getitem__(3)
    print(dataset[0])
    # gen = UNet(
    #     features=config.FEATURES,
    #     in_dim=config.INPUT_DIMENSIONS,
    #     out_dim=config.OUTPUT_DIMENSIONS
    # )

if __name__ == "__main__":
    run()
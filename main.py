import torch
from generator import Generator
import config


def run():
    gen = Generator(
        features=config.FEATURES,
        in_dim=config.INPUT_DIMENSIONS,
        out_dim=config.OUTPUT_DIMENSIONS
    )

if __name__ == "__main__":
    run()
import torch
from torch.utils.data import DataLoader

import config
from dataset import Pix2PixDataset
from generator import UNet
from discriminator import PatchGAN
from train import Training


def run():
    # Creates the training and validation dataset
    train_dataset = Pix2PixDataset(data_dir=config.TRAIN_DIR, transform=True)
    val_dataset = Pix2PixDataset(data_dir=config.VAL_DIR, transform=True)
    
    # Loads in the training and validation data
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Instantiates the generator and discriminator
    gen = UNet(
        features=config.FEATURES,
        in_dim=config.GEN_INPUT_DIMENSIONS,
        out_dim=config.GEN_OUTPUT_DIMENSIONS
    ).to(device=config.DEVICE)
    
    disc = PatchGAN(
        features=config.FEATURES,
        in_dim=config.DISC_INPUT_DIMENSIONS,
        out_dim=config.DISC_OUTPUT_DIMENSIONS
    ).to(device=config.DEVICE)
    
    # Trains the cGAN
    train_inst = Training(gen_model=gen, disc_model=disc)
    train = train_inst.cgan_training(
        epochs=config.EPOCHS,
        train_data=train_loader,
        val_data=val_loader
    )

if __name__ == "__main__":
    run()
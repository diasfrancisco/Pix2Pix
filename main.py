import torch
from torch.utils.data import DataLoader

import config
from dataset import Pix2PixDataset
from generator import UNet
from discriminator import PatchGAN
from train import Training


def run():
    train_dataset = Pix2PixDataset(data_dir=config.TRAIN_DIR, transform=True)
    val_dataset = Pix2PixDataset(data_dir=config.VAL_DIR, transform=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    gen = UNet(
        features=config.FEATURES,
        in_dim=config.INPUT_DIMENSIONS,
        out_dim=config.OUTPUT_DIMENSIONS
    ).to(device=config.DEVICE)
    
    #disc = PatchGAN().to(device=config.DEVICE)
    
    gen_train_inst = Training(model=gen)
    
    gen_train = gen_train_inst.gen_train(
        epochs=config.EPOCHS,
        data=train_loader
    )
    
    #disc_train = Training().disc_train(data=gen_train, model=disc)

if __name__ == "__main__":
    run()
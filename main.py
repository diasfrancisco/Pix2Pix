import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

import config
from dataset import Pix2PixDataset
from generator import UNet
from discriminator import PatchGAN
from utils import save_checkpoint, save_some_examples
from val import validation


def cgan_training(gen, disc, gen_optim, disc_optim, train_loader, writer, epoch):
    for batch in train_loader:
        # Train the discriminator
        x, y = batch
        x, y = x.to(device=config.DEVICE), y.to(device=config.DEVICE)

        y_gen = gen(x)
        
        # Make predictions on real and fake pairs
        real_pred = disc(x, y)
        fake_pred = disc(x, y_gen.detach())
        
        # Calculate losses of both
        disc_real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.zeros_like(real_pred))
        disc_fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        total_disc_loss = (disc_real_loss + disc_fake_loss) / 2
        
        # Compute gradients and take a step
        disc_optim.zero_grad()
        total_disc_loss.backward()
        disc_optim.step()
        
        # Train the generator
        # Make prediction on fake
        disc_fake_pred = disc(x, y_gen)
        
        # Calculate loss
        gen_fake_loss = F.binary_cross_entropy_with_logits(disc_fake_pred, torch.ones_like(disc_fake_pred))
        l1_loss = F.l1_loss(y_gen, y)
        total_gen_loss = gen_fake_loss + (l1_loss*config.LAMBDA)

        # Compute gradients and take a step
        gen_optim.zero_grad()
        total_gen_loss.backward()
        gen_optim.step()
        
        writer.add_scalars(
            'cGAN',
            {'GenLoss': total_gen_loss.item(), 'DiscLoss': total_disc_loss.item()},
            epoch
        )

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
    
    # Sets up the generator and discriminator optimisers
    gen_optim = Adam(params=gen.parameters(), lr=config.LEARNING_RATE, betas=(config.B1, config.B2))
    disc_optim = Adam(params=disc.parameters(), lr=config.LEARNING_RATE, betas=(config.B1, config.B2))
    
    writer = SummaryWriter()
    
    for epoch in range(1, config.EPOCHS+1):
        print(f'Epoch: {epoch}/{config.EPOCHS}')
        
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, gen_optim, str(config.GEN_MODEL_DIR + f'gen_{epoch}.pt'))
            save_checkpoint(disc, disc_optim, str(config.DISC_MODEL_DIR + f'disc_{epoch}.pt'))

        cgan_training(gen, disc, gen_optim, disc_optim, train_loader, writer, epoch)

    validation(val_loader, gen, gen_optim)

if __name__ == "__main__":
    run()
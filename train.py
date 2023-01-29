import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torch.optim import Adam

import config
from utils import save_checkpoint


class Training:
    def __init__(self, gen_model, disc_model):
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.transform = transforms.ToPILImage()
        self.gen_optim = Adam(params=self.gen_model.parameters(), lr=config.LEARNING_RATE, betas=(config.B1, config.B2))
        self.disc_optim = Adam(params=self.disc_model.parameters(), lr=config.LEARNING_RATE, betas=(config.B1, config.B2))
    
    def cgan_training(self, epochs, train_data, val_data):
        for epoch in range(epochs):
            if config.SAVE_MODEL and epoch % 5 == 0:
                save_checkpoint(self.gen_model.state_dict(), str(config.GEN_MODEL_DIR + config.GEN_MODEL + f'_{epoch}'))
                save_checkpoint(self.disc_model.state_dict(), str(config.DISC_MODEL_DIR + config.DISC_MODEL + f'_{epoch}'))
            
            for batch in tqdm(train_data):
                # Train the discriminator
                x, y = batch
                x, y = x.to(device=config.DEVICE), y.to(device=config.DEVICE)
                y_gen = self.gen_model(x)
                
                self.disc_optim.zero_grad()
                fake_input = torch.cat((x, y_gen.detach()), dim=1)
                real_input = torch.cat((x, y), dim=1)
                
                fake_pred = self.disc_model(fake_input)
                real_pred = self.disc_model(real_input)
                
                disc_fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
                disc_real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.zeros_like(real_pred))
                
                total_disc_loss = (disc_fake_loss + disc_real_loss) / 2
                
                total_disc_loss.backward()
                self.disc_optim.step()
                
                # Train the generator
                self.gen_optim.zero_grad()
                disc_fake_pred = self.disc_model(torch.cat((x, y_gen), dim=1))
                
                l1_loss = F.l1_loss(y_gen, y) * config.LAMBDA
                gen_fake_loss = F.binary_cross_entropy(disc_fake_pred, torch.ones_like(disc_fake_pred))
                
                total_gen_loss = gen_fake_loss + l1_loss

                total_gen_loss.backward()
                self.gen_optim.step()
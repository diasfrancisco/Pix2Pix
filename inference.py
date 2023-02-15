import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import config
from generator import UNet
from dataset import Pix2PixDataset
from utils import save_some_examples


val_dataset = Pix2PixDataset(data_dir=config.VAL_DIR, transform=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

gen = UNet(
        features=config.FEATURES,
        in_dim=config.GEN_INPUT_DIMENSIONS,
        out_dim=config.GEN_OUTPUT_DIMENSIONS
    ).to(device=config.DEVICE)

gen_optim = Adam(params=gen.parameters(), lr=config.LEARNING_RATE, betas=(config.B1, config.B2))

checkpoint = torch.load('checkpoints/15_Feb_2023_00_42_24_172485/gen/gen_140.pt')
gen.load_state_dict(checkpoint['state_dict'])
gen_optim.load_state_dict(checkpoint['optimiser'])
gen.eval()

for param_group in gen_optim.param_groups:
    param_group['lr'] = config.LEARNING_RATE
    
counter = 0
    
for batch in val_loader:
    counter += 1
    x, y = batch
    x = x.to(device=config.DEVICE)
    y = y.to(device=config.DEVICE)
    
    with torch.no_grad():
        y_gen = gen(x)
        y_gen = y_gen * 0.5 + 0.5
        save_image(y_gen, f'./test_inf/y_gen_{counter}.jpg')
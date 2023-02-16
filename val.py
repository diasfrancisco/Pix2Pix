import torch
from torchvision.utils import save_image

import config

def validation(val_loader, gen, gen_optim):
    # # For loading in a model
    # checkpoint = torch.load('checkpoints/15_Feb_2023_12_27_43_991816/gen/gen_500.pt')
    # gen.load_state_dict(checkpoint['state_dict'])
    # gen_optim.load_state_dict(checkpoint['optimiser'])
    gen.eval()
    
    # for param_group in gen_optim.param_groups:
    #     param_group['lr'] = config.LEARNING_RATE
        
    counter = 0

    for batch in val_loader:
        counter += 1
        x, y = batch
        x = x.to(device=config.DEVICE)
        y = y.to(device=config.DEVICE)
        
        with torch.no_grad():
            y_gen = gen(x)
            y_gen = y_gen * 0.5 + 0.5
            save_image(y_gen, f'{config.EXAMPLES_DIR}y_gen_{counter}.jpg')
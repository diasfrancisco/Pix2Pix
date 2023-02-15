import os
import torch
from torchvision.utils import save_image

import config


def save_checkpoint(model, optimiser, filename):
    if os.path.isdir(config.EXAMPLES_DIR):
        pass
    else:
        os.makedirs(config.EXAMPLES_DIR, exist_ok=True)
        
    # Create folders to save model checkpoints in
    if os.path.isdir(config.GEN_MODEL_DIR):
        pass
    else:
        os.makedirs(config.GEN_MODEL_DIR, exist_ok=True)
        
    if os.path.isdir(config.DISC_MODEL_DIR):
        pass
    else:
        os.makedirs(config.DISC_MODEL_DIR, exist_ok=True)
        
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimiser': optimiser.state_dict()
    }
        
    # Save my models
    torch.save(checkpoint, filename)
    
def save_some_examples(img, img_name):
    # Save PIL images
    save_image(img, config.EXAMPLES_DIR + img_name)
import os
import torch

import config


def save_checkpoint(model, filename):
    if os.path.isdir(config.BASE_DIR):
        pass
    else:
        os.mkdir(config.BASE_DIR)
        
    # Create base folders if not already present
    if os.path.isdir(config.GEN_MODEL_DIR):
        pass
    else:
        os.makedirs(config.GEN_MODEL_DIR, exist_ok=True)
        
    if os.path.isdir(config.DISC_MODEL_DIR):
        pass
    else:
        os.makedirs(config.DISC_MODEL_DIR, exist_ok=True)
        
    # Save my models
    torch.save(model, filename)
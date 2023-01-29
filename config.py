import torch
from datetime import datetime


# Architecture features
FEATURES = 64
GEN_INPUT_DIMENSIONS = 3
GEN_OUTPUT_DIMENSIONS = 3

DISC_INPUT_DIMENSIONS = 6
DISC_OUTPUT_DIMENSIONS = 1

# Directories
TRAIN_DIR = './raw/maps/train'
VAL_DIR = './raw/maps/val'

# Device type
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters for maps <-> aerial
LEARNING_RATE = 0.0002
LAMBDA = 100
B1 = 0.5
B2 = 0.999
BATCH_SIZE = 1
EPOCHS = 200

# Save and load models
GEN_MODEL_DIR = './final_models/gen/'
DISC_MODEL_DIR = './final_models/disc/'
GEN_MODEL = datetime.now().strftime('%d_%b_%Y_%H_%M_%S_%f') + '_gen'
DISC_MODEL = datetime.now().strftime('%d_%b_%Y_%H_%M_%S_%f') + '_disc'
SAVE_MODEL = 5
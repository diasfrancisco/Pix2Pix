import torch


# Architecture features
FEATURES = 64
INPUT_DIMENSIONS = 3
OUTPUT_DIMENSIONS = 3

# Directories
TRAIN_DIR = './raw/maps/train'
VAL_DIR = './raw/maps/val'

# Device type
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters for maps <-> aerial
LEARNING_RATE = 0.0002
B1 = 0.5
B2 = 0.999
BATCH_SIZE = 2
EPOCHS = 200
import torch


FEATURES = 64
INPUT_DIMENSIONS = 3
OUTPUT_DIMENSIONS = 3
ROOT_DIR = './raw/maps'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch

EPOCHS = 2
IMAGE_DIR = "input"
RANDOM_STATE = 7
IMAGE_HEIGHT = 75
IMAGE_WIDTH = 300
BATCH_SIZE = 8

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

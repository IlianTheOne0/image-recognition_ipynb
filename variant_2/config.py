import torch

RANDOM_SEED = 873912
BATCH_SIZE = 128

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4

EPOCHS = 50
PATIENCE = 7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = './data'
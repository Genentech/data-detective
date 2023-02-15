import torch

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DD_PATH = '/Users/mcconnl3/Code/data-detective/'

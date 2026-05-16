import torch
import numpy as np
import random
import os

def seed_everything(seed: int = 42):
    """
    Seed all random number generators for reproducibility.
    Sets seeds for: random, numpy, torch, and torch.cuda.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set to {seed}")

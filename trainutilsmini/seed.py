import numpy as np
import torch

def fix_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

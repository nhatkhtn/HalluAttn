import random

import numpy as np
import torch

def init_seeds(seed: int = 927):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
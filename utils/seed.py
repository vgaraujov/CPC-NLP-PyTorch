import torch
import numpy as np
import random

def set_seed(seed, use_cuda):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if use_cuda: # gpu vars
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
import random
import os
import torch
import numpy as np

def set_seed(seed, deterministic_torch, deterministic):
    """set random seed to 0 for deterministic run"""

    if deterministic:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        print("Random seed set for init and input")
        # When running on the CuDNN backend, two further options must be set

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Set Deterministic for PyTorch")
        #torch.use_deterministic_algorithms(True)
    
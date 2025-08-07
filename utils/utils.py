import torch
import numpy as np
from argparse import Namespace


def set_seeds(seed: int):
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set for random number generation.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

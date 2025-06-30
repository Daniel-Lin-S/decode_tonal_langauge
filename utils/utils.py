import torch
import numpy as np


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


def labels_to_one_hot(labels: np.ndarray) -> np.ndarray:
    """
    Turn 1d array of label indices into one-hot encoded array.
    The indices should be in the range [0, n_classes-1].

    Parameters
    ----------
    labels : np.ndarray
        1D array of label indices.
    
    Returns
    -------
    one_hot : np.ndarray
        2D array of shape (n_samples, n_classes)
        containing one-hot encoded labels.
    """
    n_classes = int(np.max(labels)) + 1
    one_hot = np.zeros((labels.shape[0], n_classes))
    one_hot[np.arange(labels.shape[0]), labels.astype(int)] = 1
    return one_hot

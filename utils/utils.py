import torch
import numpy as np
import argparse
from typing import List, Dict, Optional


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


def prepare_class_labels(
        targets: List[str],
        n_classes_dict: Optional[Dict[str, int]]=None,
        class_label_dict: Optional[Dict[str, list]]=None,
    ) -> List[str]:
    """
    Prepare class labels for the targets based on the number of classes
    and optional class label mappings.

    Parameters
    ----------
    targets : List[str]
        List of target names for which class labels are to be prepared.
    n_classes_dict : Dict[str, int]
        Dictionary mapping target names to the number of classes.
    class_label_dict : Optional[Dict[str, list]], optional
        Dictionary mapping target names to class labels, by default None.
        If not provided, class labels will be generated as strings
        from 1 to n_classes.

    Returns
    -------
    List[str]
        List of class labels for the targets. If multiple targets are provided,
        class labels will be a Cartesian product of individual target class labels.
    """
    if len(targets) > 1:
        class_labels = []
        for target in targets:
            if target not in class_label_dict or class_label_dict[target] is None:
                if n_classes_dict is None:
                    raise ValueError(
                        f"Number of classes for target '{target}' is not provided."
                    )
                # set default labels
                class_labels.append(
                    np.arange(1, n_classes_dict[target] + 1).astype(str)
                )
            else:
                class_labels.append(class_label_dict[target])

        # Generate Cartesian product of class labels for all targets
        from itertools import product
        class_labels = [
            '_'.join(label_combination)
            for label_combination in product(*class_labels)
        ]

    else:  # Handle a single target
        if n_classes_dict is None:
            raise ValueError(
                f"Number of classes for target '{targets[0]}' is not provided."
            )
        if class_label_dict[targets[0]] is None:
            class_labels = np.arange(
                1, n_classes_dict[target] + 1).astype(str)
        else:
            class_labels = class_label_dict[targets[0]]

    return class_labels

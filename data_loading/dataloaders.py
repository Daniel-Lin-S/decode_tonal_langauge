from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch


def split_dataset(
        dataset: TensorDataset, train_ratio: float,
        batch_size: int = 8,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the dataset into training and validation sets,
    and create DataLoaders for each set.

    Parameters
    ----------
    dataset : TensorDataset
        The dataset to split.
    train_ratio : float
        The ratio of the dataset to use for training.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation datasets.
    """
    # set seed
    torch.manual_seed(seed)

    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")

    n_samples = len(dataset)
    n_train = int(n_samples * train_ratio)

    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_samples - n_train]
    )

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True
    )
    test_loader = DataLoader(
        val_dataset, batch_size, shuffle=False
    )

    return train_loader, test_loader

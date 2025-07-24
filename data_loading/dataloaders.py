from typing import List
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch


def split_dataset(
        dataset: TensorDataset,
        ratios: List[float],
        shuffling: List[bool],
        batch_size: int = 8,
        seed: int = 42
    ) -> List[DataLoader]:
    """
    Splits the dataset into multiple subsets
    and create DataLoaders for each set.

    Parameters
    ----------
    dataset : TensorDataset
        The dataset to split.
    ratios : List[float]
        A list of ratios for splitting the dataset.
        Each ratio must be between 0 and 1 (exclusive).
    shuffling : List[bool]
        A list of booleans indicating whether to shuffle each subset.
        The length must match the number of ratios.
    batch_size : int, optional
        The batch size for the DataLoader, by default 8.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    List[torch.utils.data.DataLoader]
        A list of DataLoaders for each subset of the dataset.
    """
    # set seed
    torch.manual_seed(seed)

    n_samples = len(dataset)

    sample_sizes = []
    
    for i, ratio in enumerate(ratios):
        if ratio <= 0 or ratio >= 1:
            raise ValueError(
                "All ratios must be between 0 and 1 (exclusive).")

        if i == len(ratios) - 1:
            sample_sizes.append(n_samples - sum(sample_sizes))
        else:
            sample_sizes.append(int(n_samples * ratio))

    sub_datasets = random_split(
        dataset, sample_sizes
    )

    data_loaders = []

    for i, sub_dataset in enumerate(sub_datasets):
        data_loaders.append(
            DataLoader(
                sub_dataset,
                batch_size=batch_size,
                shuffle=shuffling[i]
            )
        )

    return data_loaders

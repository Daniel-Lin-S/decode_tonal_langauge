from torch.utils.data import DataLoader, TensorDataset, random_split
import torch

import os
import numpy as np
from typing import List, Optional

from .utils import match_filename


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


def collect_unlabelled_samples(
        dataset_folder: str,
        patch_size: int,
        segment_length : int,
        step_size: Optional[int]=None,
        kwords: Optional[List[str]]=None,
        verbose: bool=False
    ) -> np.ndarray:
    """
    Collect unlabelled samples of a given
    temporal length using a sliding window approach.

    Parameters
    ----------
    dataset_folder : str
        Path to the dataset folder containing `.npz` files.
    sample_length : int
        Number of timepoints in each sample.
    step_size : int
        Step size for the sliding window.
        If not specified, defaults to half of `sample_length`.
    kwords: List[str], optional
        Keywords used to filter files in the dataset_folder,
        only files with all keywords will be selected. 

    Returns
    -------
    all_samples : np.ndarray
        Array of shape (n_samples, n_channels, sample_length)
        containing all collected samples.
    """
    if step_size is None:
        step_size = segment_length // 2

    if segment_length % patch_size != 0:
        raise ValueError(
            f"segment_length ({segment_length}) must be "
            f"divisible by patch_size ({patch_size})."
        )

    n_patches = segment_length // patch_size

    all_samples = []

    # Traverse the dataset folder and its subfolders
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if not match_filename(file, 'npz', kwords):
                continue

            if file.endswith('.npz'):
                file_path = os.path.join(root, file)
                if verbose:
                    print(f"Processing file: {file_path}")

                dataset = np.load(file_path)

                try:
                    data = dataset['data']
                except KeyError:
                    raise KeyError(
                        f'Key data cannot be found in {file_path}, '
                        f'Available keys: {list(dataset.keys())}'
                    )

                # Apply sliding window
                _, n_timepoints = data.shape
                samples = []
                for start in range(0, n_timepoints - segment_length + 1, step_size):
                    end = start + segment_length
                    segment = data[:, start:end]

                    segment = segment.reshape(
                        data.shape[0], n_patches, patch_size
                    )  # (n_channels, n_patches, patch_size)

                    samples.append(segment)

                samples = np.stack(samples, axis=0)

                if verbose:
                    print('Collected {} samples with shape {}'.format(
                        len(samples), samples.shape[1:]
                    ))

                all_samples.append(samples)

    # Combine samples from all files
    all_samples = np.concatenate(all_samples, axis=0)

    if verbose:
        print('Total samples collected: ', len(all_samples))

    return all_samples

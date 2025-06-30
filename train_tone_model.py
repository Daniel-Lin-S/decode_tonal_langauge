import numpy as np
from typing import Optional, List, Dict
import warnings
import argparse


def prepare_syllable_samples(
        dataset: dict,
        tone_indices: Optional[Dict[List[int]]] = None,
        channel_indices: Optional[List[int]] = None
    ) -> np.ndarray:
    """
    Prepare samples from the dataset for training.
    
    Parameters
    ----------
    dataset : dict
        Dictionary containing the ECoG data for different tones.
        Should have keys "ECoG_toneT{tone}" where {tone} is the index of the tone,
        the values are numpy arrays of shape (n_trials, n_channels, n_timepoints).
    tone_indices : Dict[List[int]], optional
        Dictionary for filtering the tone samples. \n
        Keys: id of tones.
        Example entry:
        {1 : [0, 60], 2: [0, 60], 3 : [0, 57], 4 : [0, 58]}
    channel_indices : List[int], optional
        List of channel indices to filter the data.
        If None, all channels are used.
        Default is None.

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, n_channels, n_timepoints)
        containing the ECoG data for each syllable.
    tone_labels : np.ndarray
        Array of shape (n_samples,) containing the labels for each sample. \n
        Labels correspond to the tone index
        (starting from 0, in the order given by tone_indices)
    """
    n_tones = len([
        key for key in dataset.keys() if key.startswith("ECoG_toneT")
    ])

    samples = []
    tone_labels = []

    for tone in range(1, n_tones + 1):
        tone_key = f"ECoG_toneT{tone}"
        if tone_key not in dataset:
            warnings.warn(
                f"Key '{tone_key}' not found in dataset. Skipping tone {tone}.",
                UserWarning
            )
        
        tone_data = dataset[tone_key]

        if channel_indices is None:
            channel_indices = list(range(tone_data.shape[1]))

        start, end = tone_indices[tone] if tone_indices else (0, tone_data.shape[0])
        sample = tone_data[start:end, channel_indices, :]
        samples.append(sample)
        tone_labels.extend([tone - 1] * (end - start))

    samples = np.concatenate(samples, axis=0)
    tone_labels = np.concatenate(tone_labels, axis=0)

    return samples, tone_labels


parser = argparse.ArgumentParser(
    description="Train a tone classifier on ECoG data.")

# ----- I/O -------
parser.add_argument(
    '--data_dir', type=str, required=True,
    help='Directory containing the mat data files.')
parser.add_argument(
    '--data_file', type=str, required=True,
    help='Mat file containing the data and annotations of tones.')
parser.add_argument(
    '--figure_dir', type=str, default='figures',
    help='Directory to save the figures.'
)
# ----- Experiment Settings -------
parser.add_argument(
    '--seed', type=int, default=42,
    help='Random seed for reproducibility. Default is 42.'
)
parser.add_argument(
    '--repeat', type=int, default=1,
    help='Number of times to repeat the training. Default is 1.'
)
parser.add_argument(
    '--verbose', type=int, default=1,
    help='Verbosity level of the training process. ' \
    '0: Only the final accuracies, 1: Basic output each run (repeat), '
    '2: Detailed output for each epoch.'
)

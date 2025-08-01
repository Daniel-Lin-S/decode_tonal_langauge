from typing import List, Tuple, Dict, Optional
import numpy as np
import json


def prepare_erps_labels(
    sample_path: str,
    targets: List[str],
    channel_file: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load the ECoG dataset and prepare the ERPs and class labels for training.

    Parameters
    ----------
    sample_path : str
        Path to the npz file containing the samples.
    targets : List[str]
        List of target variables to classify, e.g. ["syllable", "tone"].
    channel_file : Optional[str], default=None
        Path to the JSON file containing channel selections for the model.
        If None, all channels will be used.

    Returns
    -------
    all_erps : np.ndarray
        The ECoG data with shape (n_samples, n_channels, n_timepoints).
    labels : np.ndarray
        The labels corresponding to the ECoG data with shape (n_samples,).
    channels : np.ndarray
        The indices of the channels used for training, based on the channel file.
    n_classes_dict : Dict[str, int]
        A dictionary mapping each target to the
        number of unique classes in that target.
    """
    dataset = np.load(sample_path)

    try:
        all_erps = dataset['ecog']
    except KeyError:
        raise KeyError(
            "The dataset does not contain 'ecog' key. "
            "Please check the data file. "
            f"Available keys in the file: {', '.join(dataset.keys())}"
        )

    target_labels = []
    n_classes_dict = {}
    for target in targets:
        if target not in dataset:
            raise KeyError(
                f"The dataset does not contain '{target}' key. "
                "Please check the data file. "
                f"Available keys in the file: {', '.join(dataset.keys())}"
            )

        target_labels.append(dataset[target].flatten())
        n_classes_dict[target] = len(np.unique(dataset[target]))

    # combine target labels into a single label array
    labels = np.zeros_like(target_labels[0], dtype=int)
    multiplier = 1
    for target_label in target_labels:
        labels += target_label * multiplier
        multiplier *= len(np.unique(target_label))

    # ------ filter channels -------
    if channel_file is not None:
        with open(channel_file, 'r') as f:
            channel_selections = json.load(f)

        channels = set()

        # Loop through all targets and union their discriminative channels
        for target in targets:
            if f'{target}_discriminative' not in channel_selections:
                raise KeyError(
                    f"Channel selection for '{target}_discriminative' "
                    f"not found in the file {channel_file}. \n"
                    "Please check the channel_file or the target variable. "
                    f"Available keys in the file: {', '.join(channel_selections.keys())}"
                )
            channels.update(channel_selections[f'{target}_discriminative'])

        # Convert the set back to a sorted list
        channels = sorted(channels)

        if len(channels) == 0:
            raise ValueError(
                f"No channels found for the targets: {', '.join(targets)}. "
                "Please check the channel file."
            )
    else:
        channels = np.arange(0, all_erps.shape[1])

    all_erps = all_erps[:, channels, :]

    return all_erps, labels, channels, n_classes_dict

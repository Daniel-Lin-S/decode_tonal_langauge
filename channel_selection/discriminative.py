"""
Module for selecting discriminative channels based on classification performance.
"""

import numpy as np
from typing import Dict, Mapping, Optional
from scipy.stats import f_oneway
import os
from typing import Optional
import matplotlib.pyplot as plt
import random

from .utils import find_significant_channels


def run(data: dict, params: dict) -> dict:
    """
    Identify discriminative channels for a specific target and return results as a dictionary.

    Args:
        data (dict): The loaded data from the `.npz` file.
        params (dict): Configuration parameters for discriminative channel selection.

    Returns:
        dict: A dictionary containing:
            - `selected_channels`: List of discriminative channels.
            - `lengths`: List of significant lengths for each channel.
            - `p_values`: List of p-values for each channel.
    """
    p_threshold = params.get('p_threshold', 0.05)
    target = params['target']

    try:
        recording_name = params.get('recording_name', 'ecog')
        ecog_sf = data[f"{recording_name}_sf"]
    except KeyError:
        raise ValueError("ECoG sampling frequency (ecog_sf) not found in the data.")

    test_results = test_discriminative_power(data, params)

    significant_channels, max_lengths = find_significant_channels(
        test_results['p_value'],
        pvalue_threshold=p_threshold,
        length_threshold=int(params["active_time_threshold"] * ecog_sf)
    )

    print(
        f'Found {len(significant_channels)} discriminative channels'
        f' for target "{target}"'
    )

    results = {
        'selected_channels': significant_channels,
        'max_lengths': max_lengths,
        'p_values': test_results['p_value']
    }

    return results


def generate_figures(
    data: dict, results: dict, params: dict, figure_dir: str
):
    os.makedirs(figure_dir, exist_ok=True)
    label_name = params['target']

    for file in os.listdir(figure_dir):
        if file.endswith('.png'):
            os.remove(os.path.join(figure_dir, file))

    sf_name = f"{params.get('recording_name', 'ecog')}_sf"

    n_channels_to_plot = min(10, len(results['selected_channels']))
    channels_to_plot = random.sample(results['selected_channels'], n_channels_to_plot)
    for ch in channels_to_plot:
        fig_name = f'{label_name}_channel_{ch}.png'
        fig_path = os.path.join(figure_dir, fig_name)

        plot_discriminative_channel(
            data, ch,
            sampling_rate=data[sf_name],
            p_vals=results['p_values'][ch, :],
            label_name=label_name,
            p_threshold=params.get('p_threshold', 0.05),
            recording_name=params['recording_name'],
            onset_time=getattr(params, 'onset_time', None),
            figure_path=fig_path,
        )

    print(f"Saved discriminative channel figures to {figure_dir}")


def test_discriminative_power(
    data: Mapping[str, np.ndarray],
    params: dict
) -> Dict[str, np.ndarray]:
    """
    Test the discriminative power of each recording channel on a given label
    using one-way ANOVA.

    Parameters
    ----------
    data : Mapping[str, np.ndarray]
        Any dictionary-like structure containing the recordings and labels.
        The recording must have shape (n_samples, n_channels, n_timepoints).
        And the labels must have shape (n_samples, 1) or (1, n_samples)
        with values being integers (e.g., categorical labels).
    params: str
        Configuration parameters including:
        - 'recording_name': Name of the recording to analyze.
        - 'label': Name of the label to use for grouping samples.


    Returns
    -------
    dict
        Dictionary containing F-statistics and p-values for each channel
        at each timepoint.
        Format: {'f_stat': np.ndarray, 'p_value': np.ndarray}
        Both arrays have shape (n_channels, n_timepoints).
    """
    recording_name = params.get('recording_name', 'ecog')
    target = params['target']

    try:
        series = data[recording_name]
    except KeyError:
        raise KeyError(
            f"Recording '{recording_name}' not found in data."
            f"Available keys: {list(data.keys())}"
        )

    if series.ndim != 3:
        raise ValueError(
            f"Recording '{recording_name}' must be a 3D array "
            "(n_samples, n_channels, n_timepoints)."
        )

    try:
        labels = data[target].squeeze()  #  (n_samples,)
    except KeyError:
        raise KeyError(
            f"Labels '{target}' not found in data."
            f"Available keys: {list(data.keys())}"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"Labels '{target}' must be a 1D array "
            "(n_samples,) or 2D array with shape (1, n_samples)"
            " or (n_samples, 1)."
        )

    if labels.shape[0] != series.shape[0]:
        raise ValueError(
            f"Number of samples in '{target}' ({labels.shape[0]}) "
            "does not match number of samples in "
            f"'{recording_name}' ({series.shape[0]})."
        )

    # Ensure labels are integers (e.g., categorical labels)
    unique_labels = np.unique(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Labels for '{target}' must be integers.")

    n_channels, n_timepoints = series.shape[1:]

    f_stat = np.zeros((n_channels, n_timepoints))
    p_value = np.zeros((n_channels, n_timepoints))

    # Perform one-way ANOVA for each channel
    for channel_idx in range(n_channels):
        channel_data = series[:, channel_idx, :]  # (n_samples, n_timepoints)

        grouped_data = [
            channel_data[labels == label, :] for label in unique_labels]

        f_result = f_oneway(*grouped_data)
        f_stat[channel_idx] = f_result.statistic   # (n_timepoints,)
        p_value[channel_idx] = f_result.pvalue   # (n_timepoints,)

    return {'f_stat': f_stat, 'p_value': p_value}


def plot_discriminative_channel(
        data : dict, channel_idx: int,
        sampling_rate: int,
        p_vals: np.ndarray,
        p_threshold: float = 0.05,
        label_name: str = 'syllable',
        recording_name: str = 'ecog',
        onset_time: Optional[int] = None,
        figure_path: Optional[str]=None
    ) -> None:
    """
    Plot the discriminative power of a specific channel over time.
    
    Parameters
    ----------
    data : dict
        Dictionary containing the recording data and labels.
        Must have keys corresponding to the recording name and label name.
    channel_idx : int
        Index of the channel to plot.
    sampling_rate : int
        Sampling rate of the recording (in Hz).
    p_vals : np.ndarray
        A one dimensional array of shape (n_timepoints, )
        with the p-values of discriminative power for the channel
        at each timepoint.
    label_name : str, optional
        Name of the label to test (default is 'syllable').
    recording_name : str, optional
        Name of the recording to test (default is 'ecog').
    onset_time : int, optional
        The time of event onset in seconds.
        (relative to the start of the recording)
    figure_path : str, optional
        Path to save the figure. If None, the figure
        will be plotted but not saved.
    """
    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First plot: Mean and SEM for each label
    series = data[recording_name]
    labels = data[label_name]

    unique_labels = np.unique(labels)

    n_timepoints = series.shape[2]

    if onset_time is not None:
        timepoints = np.arange(n_timepoints) / sampling_rate - onset_time
        timepoints = timepoints.astype(float)
    else:
        timepoints = np.arange(n_timepoints) / sampling_rate

    for label in unique_labels:
        label_data = series[labels == label, channel_idx, :]
        mean_data = np.mean(label_data, axis=0)
        std_data = np.std(label_data, axis=0)
        sem_data = std_data / np.sqrt(label_data.shape[0])

        axes[0].plot(timepoints, mean_data, label=f'{label_name} {label}')
        axes[0].fill_between(
            timepoints, mean_data - sem_data, mean_data + sem_data,
            alpha=0.2, label=f'{label_name} {label} ±1 SEM'
        )

    if onset_time is not None:
        axes[0].axvline(x=0, color='k', linestyle='--', label='Onset')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()

    # Second plot: P-values
    axes[1].plot(timepoints, p_vals, label='P-values', color='r')
    axes[1].axhline(y=p_threshold, color='k', linestyle='--',
                    label='Significance Threshold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('p-value')
    axes[1].legend()

    # set super title 
    plt.suptitle(
        f'Discriminative Power for Channel {channel_idx} '
        f'in distinguishing {label_name}', fontsize=18)

    # Save or show the figure
    if figure_path:
        plt.savefig(figure_path, dpi=500)
        plt.close()
    else:
        plt.show()

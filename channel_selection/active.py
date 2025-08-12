"""
Module for selecting active channels based on event-related responses.
"""

import numpy as np
import os
from scipy.stats import f_oneway
from typing import Optional
import matplotlib.pyplot as plt
import random

from .utils import get_max_length


def run(data: dict, params: dict) -> dict:
    """Identify active channels and return results as a dictionary."""

    erp_name = params.get('erp_name', 'ecog')
    rest_name = params.get('rest_name', 'ecog_rest')

    try:
        ecog_sf = data["ecog_sf"]
    except KeyError:
        raise ValueError("ECoG sampling frequency (ecog_sf) not found in the data.")

    length_threshold = int(params['active_time_threshold'] * ecog_sf)

    try:
        rest_samples = data[rest_name]
    except KeyError:
        raise KeyError(
            f"Recording '{rest_name}' not found in data."
            f"Available keys: {list(data.keys())}"
        )

    try:
        erp_samples = data[erp_name]
    except KeyError:
        raise KeyError(
            f"Recording '{erp_name}' not found in data."
            f"Available keys: {list(data.keys())}"
        )

    if erp_samples.shape[1:2] != rest_samples.shape[1:2]:
        raise ValueError(
            f"Shape mismatch between '{erp_name}' "
            f"and '{rest_name}': "
            f"{erp_samples.shape[1:2]} vs {rest_samples.shape[1:2]}."
        )

    n_channels = rest_samples.shape[1]
    # Bonferroni correction
    corrected_p_threshold = params['p_threshold'] / rest_samples.shape[2]  
    active_channels = []
    max_lengths = []
    p_vals_all = []

    for ch in range(n_channels):
        rest_data = rest_samples[:, ch, :]
        erp_data = erp_samples[:, ch, :]

        result = f_oneway(rest_data, erp_data)

        p_vals = result.pvalue

        significant_points = np.where(p_vals < corrected_p_threshold)[0]

        if len(significant_points) == 0:
            continue

        max_len = get_max_length(np.where(p_vals < corrected_p_threshold)[0])

        if max_len > length_threshold:
            active_channels.append(ch)
            max_lengths.append(max_len)
            p_vals_all.append(p_vals)

    print(f"Found {len(active_channels)} active channels.")

    return {
        "selected_channels": active_channels,
        "max_lengths": max_lengths,
        "p_values": p_vals,
    }


def generate_figures(
        data: dict, results: dict, params: dict, figure_dir: str
    ) -> None:
    """Generate figures for active channel selection."""

    ecog_sf = data["ecog_sf"]
    lengths = results["max_lengths"]
    channels = results["selected_channels"]
    p_vals = results["p_values"]

    # Save histogram of active lengths
    figure_path = os.path.join(figure_dir, "active_lengths.png")
    plt.figure(figsize=(10, 6))
    lengths_sec = np.array(lengths) / ecog_sf
    plt.hist(lengths_sec, bins=30, alpha=0.7, color="blue")
    plt.title("Distribution of Active Length of Significant Channels", fontsize=18)
    plt.xlabel("Active length (s)", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.savefig(figure_path, dpi=400)
    plt.close()
    print(f"Saved distribution of lengths of significant channels to {figure_path}")

    # Save ERP vs Rest plots for top channels
    n_channels_plot = min(10, len(channels))
    selected_channels = random.sample(channels, n_channels_plot)

    for i, ch in enumerate(selected_channels):
        fig_name = f"channel_{ch}_erp_rest.png"
        figure_path = os.path.join(figure_dir, fig_name)
        plot_rest_erp(
            data["ecog_rest"][:, ch, :],
            data["ecog"][:, ch, :],
            p_vals=p_vals,
            p_val_threshold=params["p_threshold"],
            sampling_rate=ecog_sf,
            figure_path=figure_path,
        )
    
    print(f"Saved ERP vs Rest plots for {n_channels_plot} channels to {figure_dir}")


def plot_rest_erp(
        rest_data: np.ndarray,
        erp_data: np.ndarray,
        p_vals: list,
        p_val_threshold: float=0.05,
        sampling_rate: int=400,
        figure_path: Optional[str]=None
    ) -> None:
    """
    Compare the activity of rest and ERP recordings for a given channel by plotting
    the mean activity ± SEM.

    Parameters
    ----------
    data : dict
        Dictionary containing the rest recording and ERP recording data.
    rest_recording_name : str
        Name of the rest recording column (e.g., 'ecog_rest').
    erp_recording_name : str
        Name of the ERP recording column (e.g., 'ecog').
    channel_idx : int
        Index of the channel to compare.
    p_vals : list
        List of p-values for the channel over time.
    p_val_threshold : float, default=0.05
        The threshold used when determining the significance of each channel.
    sampling_rate : int
        Sampling rate of the data (default: 400 Hz).
    figure_path : Optional[str], default=None
        The file path where the plot will be saved.
        If None, the plot will be displayed
        interactively instead of being saved.

    Returns
    -------
    None
    """

    if rest_data.shape[1] != erp_data.shape[1]:
        raise ValueError(
            "Rest and ERP data must have the same number of timepoints.")
    
    n_timepoints = rest_data.shape[1]

    rest_mean = rest_data.mean(axis=0)
    rest_sem = rest_data.std(axis=0) / np.sqrt(rest_data.shape[0])

    erp_mean = erp_data.mean(axis=0)
    erp_sem = erp_data.std(axis=0) / np.sqrt(erp_data.shape[0])

    time = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Rest and ERP activity
    axes[0].plot(time, rest_mean, label=f'Rest Mean ± SEM', color='blue')
    axes[0].fill_between(
        time, rest_mean - rest_sem, rest_mean + rest_sem, color='blue', alpha=0.2)

    axes[0].plot(time, erp_mean, label=f'ERP Mean ± SEM', color='orange')
    axes[0].fill_between(
        time, erp_mean - erp_sem, erp_mean + erp_sem, color='orange', alpha=0.2)

    axes[0].set_title('Comparison of Rest and ERP Activity', fontsize=16)
    axes[0].set_xlabel('Time (s)', fontsize=14)
    axes[0].set_ylabel('Amplitude', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    # Plot p-values
    axes[1].plot(time, p_vals, label='P-values', color='red')
    axes[1].axhline(
        y=p_val_threshold, color='black', linestyle='--',
        label='Significance Threshold')
    axes[1].set_title('P-values Over Time', fontsize=16)
    axes[1].set_xlabel('Time (s)', fontsize=14)
    axes[1].set_ylabel('P-value', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    # Save or show the figure
    if figure_path:
        plt.savefig(figure_path, dpi=400, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

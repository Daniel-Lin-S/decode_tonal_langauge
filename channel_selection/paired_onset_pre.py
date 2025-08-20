"""Channel selection based on paired speech onset and pre-onset signals."""

import numpy as np
from typing import Optional
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import os
import random

from .utils import get_max_length


def run(data: dict, params: dict) -> dict:
    """Identify channels with significant onset vs pre-onset difference."""
    onset_name = params.get('onset_name', 'ecog_onset')
    pre_name = params.get('pre_name', 'ecog_pre')

    try:
        ecog_sf = data['ecog_sf']
    except KeyError:
        raise ValueError('ECoG sampling frequency (ecog_sf) not found in data.')

    try:
        onset_samples = data[onset_name]
        pre_samples = data[pre_name]
    except KeyError as e:
        raise KeyError(
            f"Recording '{e.args[0]}' not found in data. Available keys: {list(data.keys())}")

    if onset_samples.shape != pre_samples.shape:
        raise ValueError(
            f"Shape mismatch between '{onset_name}' and '{pre_name}': "
            f"{onset_samples.shape} vs {pre_samples.shape}.")

    n_channels = onset_samples.shape[1]
    corrected_p = params['p_threshold'] / onset_samples.shape[2]
    length_threshold = int(params['active_time_threshold'] * ecog_sf)

    selected = []
    lengths = []
    p_all = []

    for ch in range(n_channels):
        onset_data = onset_samples[:, ch, :]
        pre_data = pre_samples[:, ch, :]
        _, p_vals = ttest_rel(onset_data, pre_data, axis=0)
        p_all.append(p_vals)
        significant = np.where(p_vals < corrected_p)[0]
        if len(significant) == 0:
            continue
        max_len = get_max_length(significant)
        if max_len > length_threshold:
            selected.append(ch)
            lengths.append(max_len)

    print(f"Found {len(selected)} active channels.")
    return {
        'selected_channels': selected,
        'max_lengths': lengths,
        'p_values': p_all,
    }


def generate_figures(data: dict, results: dict, params: dict, figure_dir: str) -> None:
    ecog_sf = data['ecog_sf']
    lengths = results['max_lengths']
    channels = results['selected_channels']
    p_vals = results['p_values']

    figure_path = os.path.join(figure_dir, 'active_lengths.png')
    plt.figure(figsize=(10,6))
    lengths_sec = np.array(lengths) / ecog_sf
    if len(lengths_sec) > 0:
        plt.hist(lengths_sec, bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of Active Length of Significant Channels', fontsize=18)
    plt.xlabel('Active length (s)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.savefig(figure_path, dpi=400)
    plt.close()
    print(f"Saved distribution of lengths of significant channels to {figure_path}")

    n_channels_plot = min(10, len(channels))
    if n_channels_plot == 0:
        return
    selected_channels = random.sample(channels, n_channels_plot)
    for ch in selected_channels:
        fig_name = f"channel_{ch}_pre_onset.png"
        path = os.path.join(figure_dir, fig_name)
        plot_pre_onset(
            data[params.get('pre_name', 'ecog_pre')][:, ch, :],
            data[params.get('onset_name', 'ecog_onset')][:, ch, :],
            p_vals=p_vals[ch],
            p_val_threshold=params['p_threshold'],
            sampling_rate=ecog_sf,
            figure_path=path,
        )


def plot_pre_onset(
        pre_data: np.ndarray,
        onset_data: np.ndarray,
        p_vals: np.ndarray,
        p_val_threshold: float = 0.05,
        sampling_rate: int = 400,
        figure_path: Optional[str] = None,
    ) -> None:
    """Plot pre-onset vs onset activity for a channel."""

    if pre_data.shape[1] != onset_data.shape[1]:
        raise ValueError('Pre-onset and onset data must have same number of timepoints.')

    n_timepoints = pre_data.shape[1]
    pre_mean = pre_data.mean(axis=0)
    pre_sem = pre_data.std(axis=0) / np.sqrt(pre_data.shape[0])
    onset_mean = onset_data.mean(axis=0)
    onset_sem = onset_data.std(axis=0) / np.sqrt(onset_data.shape[0])
    time = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)

    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    axes[0].plot(time, pre_mean, label='Pre-onset Mean ± SEM', color='blue')
    axes[0].fill_between(time, pre_mean - pre_sem, pre_mean + pre_sem, color='blue', alpha=0.2)
    axes[0].plot(time, onset_mean, label='Onset Mean ± SEM', color='orange')
    axes[0].fill_between(time, onset_mean - onset_sem, onset_mean + onset_sem, color='orange', alpha=0.2)
    axes[0].set_title('Comparison of Pre-onset and Onset Activity', fontsize=16)
    axes[0].set_xlabel('Time (s)', fontsize=14)
    axes[0].set_ylabel('Amplitude', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time, p_vals, label='P-values', color='red')
    axes[1].axhline(y=p_val_threshold, color='black', linestyle='--')
    axes[1].set_title('Paired t-test p-values', fontsize=16)
    axes[1].set_xlabel('Time (s)', fontsize=14)
    axes[1].set_ylabel('p-value', fontsize=14)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)

    if figure_path is None:
        plt.show()
    else:
        plt.savefig(figure_path, dpi=400)
        plt.close()

"""
Find and visualise discriminative channels in ECoG data using YAML configuration.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat

from data_loading.channel_selection import (
    test_discriminative_power,
    find_significant_channels,
)
from utils.visualise import plot_discriminative_channel
from utils.config import dict_to_namespace


def run(config: dict, config_path: str | None = None) -> None:
    """Identify discriminative channels based on configuration."""

    ch_cfg = config.get("channel_selection_discriminative", {}).get("params", {})
    params_dict = {}
    for section in ("io", "experiment", "settings", "training"):
        params_dict.update(ch_cfg.get(section, {}))
    params = dict_to_namespace(params_dict)

    if len(params.p_thresholds) != len(params.label_names):
        raise ValueError('Number of p-value thresholds must match the number of label names.')
    if len(params.consecutive_length_thresholds) != len(params.label_names):
        raise ValueError('Number of consecutive length thresholds must match the number of label names.')

    data = np.load(params.recording_file_path)

    location_data = loadmat(params.channel_locations_file)
    try:
        channel_locations = location_data[params.channel_locations_key]
    except KeyError as e:
        raise KeyError(
            f"Key '{params.channel_locations_key}' not found in the file '{params.channel_locations_file}'."
        ) from e

    channel_data = {}
    p_vals = {}
    discriminative_powers = {}

    for i, label_name in enumerate(params.label_names):
        results = test_discriminative_power(
            data, label_name, recording_name=params.recording_name
        )
        discriminative_scores = -np.log10(results['p_value'] + 1e-10)
        threshold = -np.log10(0.05)
        auc_scores = np.sum(
            np.clip(discriminative_scores - threshold, 0, None), axis=1
        )
        significant_channels = find_significant_channels(
            results,
            pvalue_threshold=params.p_thresholds[i],
            consecutive_length_threshold=params.consecutive_length_thresholds[i],
        )
        print(
            f'Found {len(significant_channels)} discriminative channels for label "{label_name}"'
        )
        channel_data[f'{label_name}_discriminative'] = significant_channels
        p_vals[label_name] = results['p_value'][significant_channels, :]
        discriminative_powers[label_name] = auc_scores

    if getattr(params, "channel_output_file", None):
        df = pd.DataFrame({
            'x': channel_locations[:, 0],
            'y': channel_locations[:, 1],
            'z': channel_locations[:, 2],
        })
        for label_name in params.label_names:
            if len(discriminative_powers[label_name]) != len(channel_locations):
                raise ValueError(
                    f"Discriminative powers for label '{label_name}' do not match number of electrodes."
                )
            df[f'{label_name}_discriminative_score'] = discriminative_powers[label_name]
        df.to_csv(params.channel_output_file, index=True)
        print('Saved channel locations and discriminative scores to', params.channel_output_file)

    if getattr(params, "output_file", None):
        output_dir = os.path.dirname(params.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if os.path.exists(params.output_file):
            with open(params.output_file, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
            existing_data.update(channel_data)
            with open(params.output_file, 'w') as f:
                json.dump(existing_data, f, indent=4)
            print('Appended discriminative channels to', params.output_file)
        else:
            with open(params.output_file, 'w') as f:
                json.dump(channel_data, f, indent=4)
            print('Saved discriminative channels to', params.output_file)

    if getattr(params, "figure_dir", None):
        if not os.path.exists(params.figure_dir):
            os.makedirs(params.figure_dir)
        for file in os.listdir(params.figure_dir):
            if file.endswith('.png'):
                os.remove(os.path.join(params.figure_dir, file))

        if getattr(params, "individual_figures", False):
            for i, significant_channels in enumerate(channel_data.values()):
                label_name = params.label_names[i]
                for ch in significant_channels:
                    fig_name = f'{label_name}_channel_{ch}.png'
                    fig_path = os.path.join(params.figure_dir, fig_name)
                    plot_discriminative_channel(
                        data,
                        params.recording_name,
                        ch,
                        p_vals[label_name][list(channel_data.values())[i] == ch][0]
                        if len(p_vals[label_name]) > i else None,
                        params.sampling_rate,
                        params.onset_time,
                        fig_path,
                    )
                    print('Saved figure to', fig_path)
        else:
            summary_fig = os.path.join(params.figure_dir, 'discriminative_channels.png')
            plot_discriminative_channel(
                data,
                params.recording_name,
                channel_data,
                p_vals,
                params.sampling_rate,
                params.onset_time,
                summary_fig,
            )
            print('Saved summary figure to', summary_fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python find_discriminative_channels.py <config.yaml>")
    from utils.config import load_config
    cfg = load_config(sys.argv[1])
    run(cfg, sys.argv[1])

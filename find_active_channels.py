"""
Find the channels with responses at events compared to rest period.
Parameters are provided via a YAML configuration.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from data_loading.channel_selection import find_active_channels
from utils.visualise import plot_rest_erp
from utils.config import dict_to_namespace


def run(config: dict, config_path: str | None = None) -> None:
    """Identify active channels using configuration settings."""

    ch_cfg = config.get("channel_selection_active", {}).get("params", {})
    params_dict = {}
    for section in ("io", "experiment", "settings", "training"):
        params_dict.update(ch_cfg.get(section, {}))
    params = dict_to_namespace(params_dict)

    data = np.load(params.recording_file_path)

    channels, lengths, p_vals = find_active_channels(
        data,
        params.rest_recording_name,
        params.erp_recording_name,
        params.p_threshold,
        params.consecutive_length_threshold,
    )

    print(f'Found {len(channels)} active channels')

    if getattr(params, "output_file", None):
        output_data = {"active_channels": channels}
        configs_dir = os.path.dirname(params.output_file)
        if not os.path.exists(configs_dir):
            os.makedirs(configs_dir)
        if os.path.exists(params.output_file):
            with open(params.output_file, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
            existing_data.update(output_data)
            with open(params.output_file, "w") as f:
                json.dump(existing_data, f, indent=4)
            print('Appended active channels to', params.output_file)
        else:
            with open(params.output_file, "w") as f:
                json.dump(output_data, f, indent=4)
            print('Saved active channels to', params.output_file)

    if getattr(params, "figure_dir", None):
        if not os.path.exists(params.figure_dir):
            os.makedirs(params.figure_dir)
        figure_path = os.path.join(params.figure_dir, 'active_channels_length.png')
        plt.figure(figsize=(10, 6))
        lengths_sec = np.array(lengths) / params.sampling_rate
        plt.hist(lengths_sec, bins=30, alpha=0.7, color='blue')
        plt.title('Distribution of Active Length of Significant Channels', fontsize=18)
        plt.xlabel('Active length (s)', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.savefig(figure_path, dpi=400)
        plt.close()
        print('Saved distribution of lengths of significant channels to ', figure_path)

        n_channels_plot = min(10, len(channels))
        for i, ch in enumerate(channels[:n_channels_plot]):
            fig_name = f'channel_{ch}_erp_rest.png'
            figure_path = os.path.join(params.figure_dir, fig_name)
            plot_rest_erp(
                data,
                params.rest_recording_name,
                params.erp_recording_name,
                ch,
                p_vals=p_vals[i],
                p_val_threshold=params.p_threshold,
                sampling_rate=params.sampling_rate,
                figure_path=figure_path,
            )
            print('Saved ERP plot for channel', ch, 'to', figure_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python find_active_channels.py <config.yaml>")
    from utils.config import load_config
    cfg = load_config(sys.argv[1])
    run(cfg, sys.argv[1])

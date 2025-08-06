"""
Find the channels with responses at events compared to rest period.
Parameters are provided via a YAML configuration.
"""

import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace

from data_loading.channel_selection import find_active_channels
from utils.visualise import plot_rest_erp
from utils.config import (
    dict_to_namespace, load_config, append_data_json,
    update_configuration
)


def run(config: dict) -> None:
    """Identify active channels using configuration settings."""

    ch_cfg = config.get("channel_selection_active", {}).get("params", {})
    params_dict = {}
    for section in ("io", "settings"):
        params_dict.update(ch_cfg.get(section, {}))

    params = dict_to_namespace(params_dict)

    output_dir_name = generate_output_dir_name(
        os.path.basename(params.sample_dir), params
    )
    output_dir = os.path.join(params.output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    update_configuration(
        output_path = os.path.join(output_dir, "config.yaml"),
        previous_config_path = os.path.join(params.sample_dir, "config.yaml"),
        new_module='channel_selection_active',
        new_module_cfg=config.get("channel_selection_active", {})
    )

    for file_name in os.listdir(params.sample_dir):
        if not file_name.endswith(".npz") or not file_name.startswith("subject_"):
            continue

        subject_id = file_name.split("_")[1].split(".")[0]
        sample_file_path = os.path.join(params.sample_dir, file_name)
        data = np.load(sample_file_path)
        
        try:
            ecog_sf = data['ecog_sf']
        except KeyError:
            raise ValueError(
                f"ECoG sampling frequency (ecog_sf) not found in {params.recording_file_path}."
            )

        channels, lengths, p_vals = find_active_channels(
            data,
            p_val_threshold=params.p_threshold,
            length_threshold=params.consecutive_length_threshold,
        )

        print(f'Found {len(channels)} active channels for subject {subject_id}.')

        output_file = os.path.join(output_dir, f'subject_{subject_id}.json')
        output_data = {"active_channels": channels}

        append_data_json(output_file, output_data)

        if getattr(params, "figure_dir", None):
            subject_figure_dir = os.path.join(params.figure_dir, f'subject_{subject_id}')
            os.makedirs(subject_figure_dir, exist_ok=True)

            figure_path = os.path.join(subject_figure_dir, f'active_lengths.png')
            plt.figure(figsize=(10, 6))
            lengths_sec = np.array(lengths) / ecog_sf
            plt.hist(lengths_sec, bins=30, alpha=0.7, color='blue')
            plt.title(
                f'Subject {subject_id}: Distribution of Active Length of Significant Channels',
                fontsize=18
            )
            plt.xlabel('Active length (s)', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.savefig(figure_path, dpi=400)
            plt.close()
            print('Saved distribution of lengths of significant channels to ', figure_path)

            n_channels_plot = min(10, len(channels))
            for i, ch in enumerate(channels[:n_channels_plot]):
                fig_name = f'channel_{ch}_erp_rest.png'
                figure_path = os.path.join(subject_figure_dir, fig_name)
                plot_rest_erp(
                    data['ecog_rest'][:, ch, :],
                    data['ecog'][:, ch, :],
                    p_vals=p_vals[i],
                    p_val_threshold=params.p_threshold,
                    sampling_rate=ecog_sf,
                    figure_path=figure_path,
                )
                print('Saved ERP plot for channel', ch, 'to', figure_path)


def generate_output_dir_name(
        base_name: str, params: Namespace
    ) -> str:
    """
    Generate a unique and human-readable name for the output directory
    based on the sample directory and channel selection parameters.
    """
    hash_input = f"p_threshold={params.p_threshold}_length_threshold={params.consecutive_length_threshold}"
    hash_part = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"{base_name}__{hash_part}"


if __name__ == "__main__":
    import sys
    from utils.config import load_config

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python find_active_channels.py <config.yaml>")

    cfg = load_config(sys.argv[1])
    run(cfg)

"""
Align ECoG and audio samples using TextGrid annotations.
Configuration is provided via YAML.
"""

import os
import yaml
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from data_loading.text_align import handle_textgrids, extract_ecog_audio
from utils.config import dict_to_namespace, update_configuration


def run(config: dict) -> None:
    """Extract samples from multiple subjects based on configuration."""

    collection_cfg = config.get("sample_collection", {})
    params_config = collection_cfg.get("params", {})
    params_dict = {}
    for section in ("io", "settings"):
        params_dict.update(params_config.get(section, {}))
    params = dict_to_namespace(params_dict)

    if not hasattr(params, "overwrite"):
        params.overwrite = False

    output_dir_name = _generate_output_dir_name(
        os.path.basename(params.recording_dir),
        collection_cfg
    )

    output_dir = os.path.join(params.output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    figure_root = os.path.join(output_dir, 'figures')
    os.makedirs(figure_root, exist_ok=True)

    update_configuration(
        output_path=os.path.join(output_dir, "config.yaml"),
        previous_config_path=os.path.join(params.recording_dir, "config.yaml"),
        new_module='sample_collection',
        new_module_cfg=collection_cfg
    )

    for subject_id, subject_params in params_config.get("subjects", {}).items():
        subject_path = os.path.join(params.recording_dir, f"subject_{subject_id}")
        if not os.path.exists(subject_path):
            print(
                f"Recording directory {subject_path} not found. Skipping..."
            )
            continue

        subject_output_path = os.path.join(
            output_dir, f"subject_{subject_id}.npz"
        )
        if os.path.exists(subject_output_path) and not params.overwrite:
            print(f"Output file {subject_output_path} already exists. Skipping ...")
            continue

        textgrid_dir = os.path.join(params.textgrid_root, subject_params["textgrid_dir"])
        if not os.path.exists(textgrid_dir):
            print(f"TextGrid directory {textgrid_dir} not found. Skipping...")
            continue

        print(
            '------------------------ \n'
            f'Extracting all samples from {subject_path} using textgrids from {textgrid_dir}'
            '\n ------------------------'
        )

        intervals = handle_textgrids(
            textgrid_dir,
            start_offset=subject_params.get("start_offset", 0.0),
            tier_list=subject_params.get("tier_list", None),
            blocks=subject_params.get("blocks", None),
        )

        if len(intervals) == 0:
            raise ValueError(
                "No intervals found in the TextGrid files. "
                "Check the directory and file naming conventions."
                f"Target blocks: {params.blocks if params.blocks else 'all'}"
            )

        print(
            "Extracted intervals from TextGrid files: "
            f"{len(intervals)} blocks found."
        )

        if intervals:
            for block_id, block_df in intervals.items():
                if not block_df.empty:
                    events = block_df.to_dict('records')
                    sampled_events = _sample_consecutive_events(events, num_events=3)

                    ecog_path = os.path.join(subject_path, f"B{block_id}_ecog.npz")

                    if os.path.exists(ecog_path):
                        ecog = np.load(ecog_path)
                        signal = ecog['data']
                        sf = int(ecog['sf'])
                        channels = np.random.choice(
                            signal.shape[0], size=5, replace=False)
                        fig_dir = os.path.join(figure_root, f'subject_{subject_id}')
                        os.makedirs(fig_dir, exist_ok=True)

                        plot_ecog_events(
                            signal, sf, sampled_events, channels,
                            subject_id, block_id, fig_dir
                        )

        extract_ecog_audio(
            intervals,
            subject_path,
            syllables=params.syllable_identifiers,
            length=subject_params["sample_length"],
            output_path=subject_output_path,
            rest_period=tuple(subject_params["rest_period"]),
        )


def _sample_consecutive_events(events, num_events):
    events = sorted(events, key=lambda x: x['start'])

    if len(events) > num_events:
        start_idx = np.random.randint(0, len(events) - num_events + 1)
        return events[start_idx:start_idx + num_events]
    else:
        return events



def _generate_output_dir_name(base_name: str, collection_cfg: dict) -> str:
    """
    Generate a unique and human-readable name for the output directory
    based on the recording directory and sample extraction parameters.
    """
    hash_input = yaml.dump(collection_cfg, sort_keys=True)
    hash_part = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"{base_name}__{hash_part}"


def plot_ecog_events(
    signal: np.ndarray,
    sf: int,
    events: list,
    channels: list,
    subject_id: str,
    block_id: str,
    fig_dir: str
) -> None:
    """
    Plot ECoG signal for multiple channels, with each channel occupying a subplot.
    Highlight events and include inter-event signals.

    Args:
        signal (np.ndarray): ECoG signal (channels x timepoints).
        sf (int): Sampling frequency of the signal.
        events (list): List of event intervals, each as a dict with 'start' and 'end'.
        channels (list): List of channel indices to plot.
        subject_id (str): Subject ID for labeling.
        block_id (str): Block ID for labeling.
        fig_dir (str): Directory to save the figure.
    """
    os.makedirs(fig_dir, exist_ok=True)

    start_time = max(min(event['start'] for event in events) - 0.5, 0)
    end_time = max(event['end'] for event in events) + 0.5
    start_idx = int(start_time * sf)
    end_idx = int(end_time * sf)
    time = np.arange(start_idx, end_idx) / sf

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 4 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for ax, ch_idx in zip(axes, channels):
        ax.plot(
            time,
            signal[ch_idx, start_idx:end_idx],
            label=f'Offset',
            color='blue',
            alpha=0.7
        )

        for i, event in enumerate(events):
            event_start_idx = int(event['start'] * sf)
            event_end_idx = int(event['end'] * sf)
            event_time = np.arange(event_start_idx, event_end_idx) / sf

            ax.plot(
                event_time,
                signal[ch_idx, event_start_idx:event_end_idx],
                label=f'Onset' if i == 0 else None,
                color='orange'
            )
            ax.axvline(
                event['start'], color='g',
                linestyle='--', alpha=0.7,
                label='Event Start' if i == 0 else None
            )
            ax.axvline(
                event['end'], color='r',
                linestyle='--', alpha=0.7,
                label='Event End' if i == 0 else None
            )

        ax.set_title(f'Channel {ch_idx}', fontsize=18)
        ax.set_ylabel('Amplitude', fontsize=16)

        ax.legend(
            fontsize=14, loc='upper right',
            bbox_to_anchor=(1.2, 1),
            borderaxespad=0.
        )

    axes[-1].set_xlabel('Time (s)', fontsize=16)
    fig.suptitle(f'Subject {subject_id} Block {block_id}', fontsize=20)
    fig.tight_layout()

    fig.subplots_adjust(top=0.93)

    fig_path = os.path.join(fig_dir, f'block_{block_id}_events.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    from utils.config import load_config

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python extract_samples.py <config.yaml>")
    
    cfg = load_config(sys.argv[1])
    run(cfg)

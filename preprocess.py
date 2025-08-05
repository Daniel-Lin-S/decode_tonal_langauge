"""
Extract ECoG signal from TDT blocks, preprocess, and save to .npz files.

This script now reads its parameters from a YAML configuration file.
"""

import os
import tdt
import numpy as np

from data_loading.preprocessing import (
    hilbert_filter,
    downsample,
    zscore,
    rereference,
    bandpass_filter,
)
from utils.config import dict_to_namespace


def run(config: dict, config_path: str | None = None) -> None:
    """Extract and preprocess ECoG signals based on configuration."""

    pre_cfg = config.get("preprocess", {}).get("params", {})
    params_dict = {}
    for section in ("io", "experiment", "settings", "training"):
        params_dict.update(pre_cfg.get(section, {}))
    params = dict_to_namespace(params_dict)

    freq_ranges = getattr(params, "freq_ranges", None)
    freq_band = getattr(params, "freq_band", None)
    freq = getattr(params, "downsample_freq", 400)

    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    for dir in os.listdir(params.tdt_dir):
        try:
            block_id = int(dir.split('-')[-1].replace('B', ''))
            print(f'Processing block {block_id} of subject {params.subject_id}...')
        except ValueError:
            print(
                f"Skipping directory '{dir}' as it does not match expected format.",
                "Expected format: 'HS<subject_id>-<block_id>'.",
            )
            continue

        if freq_ranges is not None:
            if freq_band is None:
                raise ValueError(
                    "freq_band must be specified when freq_ranges is provided."
                )
            file_name = f'HS{params.subject_id}_B{block_id}_ecog_{freq_band}_{freq}Hz.npz'
        else:
            file_name = f'HS{params.subject_id}_B{block_id}_ecog_{freq}Hz.npz'

        ecog_file = os.path.join(params.output_dir, file_name)
        audio_file = os.path.join(
            params.output_dir, f'HS{params.subject_id}_B{block_id}_sound.npz'
        )

        if os.path.exists(ecog_file) and os.path.exists(audio_file):
            print(f'Skipping block {block_id}, already processed.')
            continue

        block_path = os.path.join(params.tdt_dir, dir)
        block_data = tdt.read_block(block_path)

        data = block_data.streams.EOG1.data
        ecog_freq = block_data.streams.EOG1.fs
        audio = block_data.streams.ANIN.data
        audio_freq = block_data.streams.ANIN.fs

        print('Audio shape: ', audio.shape)
        print('ECoG data shape: ', data.shape)
        print('ECoG sampling frequency:', ecog_freq)

        audio = audio[:1, :]   # mono-channel audio
        ecog_down = downsample(data, ecog_freq, freq)

        if freq_ranges is not None:
            all_channels = []
            for freq_range in freq_ranges:
                if len(freq_range) != 2:
                    raise ValueError(
                        "Each frequency range must have exactly two elements."
                    )
                if getattr(params, "envelope", False):
                    signals = hilbert_filter(
                        ecog_down, freq, freq_ranges=[freq_range]
                    )
                else:
                    signals = bandpass_filter(
                        ecog_down,
                        lowcut=freq_range[0],
                        highcut=freq_range[1],
                        fs=freq,
                    )
                all_channels.append(signals)
            ecog_filtered = np.concatenate(all_channels, axis=0)
        else:
            ecog_filtered = ecog_down

        if params.normalisation == 'zscore':
            ecog_normalised = zscore(ecog_filtered)
        elif params.normalisation == 'rereference':
            start, end = params.rereference_interval
            start_sample = int(start * freq)
            end_sample = int(end * freq)
            ecog_normalised = rereference(
                ecog_filtered, (start_sample, end_sample)
            )
        else:
            raise ValueError("Invalid normalisation method specified.")

        if not os.path.exists(ecog_file):
            np.savez(ecog_file, data=ecog_normalised, sf=freq)
            print('Saved ECoG data to:', ecog_file)
        else:
            print('ECoG data already exists:', ecog_file)

        if not os.path.exists(audio_file):
            np.savez(audio_file, data=audio, sf=int(audio_freq))
            print('Saved audio data to:', audio_file)
        else:
            print('Audio data already exists:', audio_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python preprocess.py <config.yaml>")
    from utils.config import load_config
    cfg = load_config(sys.argv[1])
    run(cfg, sys.argv[1])

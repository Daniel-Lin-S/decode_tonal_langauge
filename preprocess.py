"""
Extract ECoG signal from TDT blocks, preprocess, and save to .npz files.

Preprocessing pipeline:
1. Downsample the ECoG data to a target frequency. (e.g. 400Hz)
2. Apply Hilbert filter to extract high gamma activity.
3. Normalise the ECoG data using either z-score or rereferencing.

Required hyper-parameters from the JSON file:
- freq_ranges: a list of lists, each list should have length 2 indicating the
  lower and higher bounds of the frequency band, e.g. [[0.3, 17], [70, 150]]
  Warning: using more than one frequency band results in more channels,
  each band is extracted by a different filter and z-scored separately.
  e.g. 2 bands gives double number of channels.
  Note: if not provided, Hilbert filter will not be applied.
- freq_band: str, a marker for the freq_ranges, will be used to identify the .npz
  files extracted.
"""


import argparse
import os
import tdt
from data_loading.preprocessing import (
    hilbert_filter, downsample,
    zscore, rereference,
    bandpass_filter
)
import json
import numpy as np


parser = argparse.ArgumentParser(
    description="Extract ECoG signal from TDT blocks, preprocess, and save to .npz files."
)

# ------- I/O ---------
parser.add_argument(
    '--tdt_dir', default='raw/ecog', type=str,
    help='Directory containing TDT blocks. '
    'Each block should be in a separate subdirectory.'
    'Each subdirectory should be named as "HS<subject_id>-<block_id>".'
    'The data should have streams EOG1 and ANIN for ECoG and audio respectively.'
)
parser.add_argument(
    '--output_dir', default='processed/npz', type=str,
    help='Directory to save the processed ECoG data in .npz format.'
)
parser.add_argument(
    '--subject_id', required=True, type=int,
    help='Subject ID, used to name the output files.'
)
parser.add_argument(
    '--config_file', required=True, type=str,
    help='Path to the JSON file with necessary hyperparameters'
)
# ------- Preprocess Settings ---------
parser.add_argument(
    '--normalisation', default='rereference', type=str,
    choices=['zscore', 'rereference'],
    help='Normalisation method to apply to the ECoG data.'
    'zscore: Z-score normalisation across the whole recording.'
    'rereference: normalise using a specified interval. (e.g. resting period)'
)
parser.add_argument(
    '--rereference_interval', default=(0, 1), type=float, nargs=2,
    help='Interval (in seconds) for rereferencing the ECoG data.'
)
parser.add_argument(
    '--envelope', action='store_true',
    help='If set, apply Hilbert filter to extract high gamma activity.'
)


args = parser.parse_args()

with open(args.config_file, 'r') as f:
    config = json.load(f)

freq_ranges = config.get('freq_ranges', None)
freq_band = config.get('freq_band', None)
freq = config.get('downsample_freq', 400)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for dir in os.listdir(args.tdt_dir):
    try:
        block_id = int(dir.split('-')[-1].replace('B', ''))
        print(f'Processing block {block_id} of subject {args.subject_id}...')
    except ValueError:
        print(
            f"Skipping directory '{dir}' as it does not match expected format."
            "Expected format: 'HS<subject_id>-<block_id>'."
        )
        continue

    if freq_ranges is not None:
        if freq_band is None:
            raise ValueError(
                "freq_band must be specified when freq_ranges is provided."
            )
        file_name = f'HS{args.subject_id}_B{block_id}_ecog_{freq_band}_{freq}Hz.npz'
    else:
        file_name = f'HS{args.subject_id}_B{block_id}_ecog_{freq}Hz.npz'

    ecog_file = os.path.join(args.output_dir, file_name)

    audio_file = os.path.join(
        args.output_dir, f'HS{args.subject_id}_B{block_id}_sound.npz'
    )

    if os.path.exists(ecog_file) and os.path.exists(audio_file):
        print(f'Skipping block {block_id}, already processed.')
        continue

    block_path = os.path.join(args.tdt_dir, dir)

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

            if args.envelope:
                signals = hilbert_filter(
                    ecog_down, freq, freq_ranges=[freq_range]
                )
            else:
                signals = bandpass_filter(
                    ecog_down,
                    lowcut=freq_range[0], highcut=freq_range[1],
                    fs=freq
                )
            all_channels.append(signals)

        ecog_filtered = np.concatenate(
            all_channels, axis=0)  # concatenate in channel dimension

    else:
        ecog_filtered = ecog_down

    if args.normalisation == 'zscore':
        ecog_normalised = zscore(ecog_filtered)
    elif args.normalisation == 'rereference':
        start, end = args.rereference_interval
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

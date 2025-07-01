"""
Extract ECoG signal from TDT blocks, preprocess, and save to .mat files.

Preprocessing pipeline:
1. Downsample the ECoG data to a target frequency. (e.g. 400Hz)
2. Apply Hilbert filter to extract high gamma activity.
3. Normalise the ECoG data using either z-score or rereferencing.
"""


import argparse
import os
import tdt
from data_loading.preprocessing import (
    hilbert_filter, downsample,
    zscore, rereference
)
from scipy.io import savemat


parser = argparse.ArgumentParser(
    description="Extract ECoG signal from TDT blocks, preprocess, and save to .mat files."
)

parser.add_argument(
    '--subject_id', required=True, type=int,
    help='Subject ID for which to process the data.'
)
parser.add_argument(
    '--tdt_dir', default='raw/ecog', type=str,
    help='Directory containing TDT blocks. '
    'Each block should be in a separate subdirectory.'
    'Each subdirectory should be named as "HS<subject_id>-<block_id>".'
    'The data should have streams EOG1 and ANIN for ECoG and audio respectively.'
)
parser.add_argument(
    '--output_dir', default='processed/mat', type=str,
    help='Directory to save the processed ECoG data in .mat format.'
)
parser.add_argument(
    '--downsample_freq', default=400, type=int,
    help='Target sampling frequency of ECoG data after downsampling.'
)
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


# TODO turn this into config file
freq_ranges = (70, 150)  # High gamma frequency range
freq_band = 'hga'

args = parser.parse_args()

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

    ecog_file = os.path.join(
        args.output_dir,
        f'HS{args.subject_id}_B{block_id}_ecog_{freq_band}_{args.downsample_freq}Hz.mat'
    )

    audio_file = os.path.join(
        args.output_dir, f'HS{args.subject_id}_B{block_id}_sound.mat'
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

    ecog_down = downsample(data, ecog_freq, args.downsample_freq)

    ecog_filtered = hilbert_filter(
        ecog_down, args.downsample_freq, freq_ranges=freq_ranges)

    if args.normalisation == 'zscore':
        ecog_normalised = zscore(ecog_filtered)
    elif args.normalisation == 'rereference':
        start, end = args.rereference_interval
        start_sample = int(start * args.downsample_freq)
        end_sample = int(end * args.downsample_freq)
        ecog_normalised = rereference(
            ecog_filtered, (start_sample, end_sample))
    else:
        raise ValueError("Invalid normalisation method specified.")

    if not os.path.exists(ecog_file):
        savemat(ecog_file, {
            'data': ecog_normalised,
            'sf': args.downsample_freq,
        })
        print('Saved ECoG data to:', ecog_file)
    else:
        print('ECoG data already exists:', ecog_file)

    if not os.path.exists(audio_file):
        savemat(audio_file, {
            'data': audio,
            'sf': int(audio_freq),
        })
        print('Saved audio data to:', audio_file)
    else:
        print('Audio data already exists:', audio_file)

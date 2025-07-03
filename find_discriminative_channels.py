import os
import argparse
from scipy.io import loadmat
import json

from data_loading.channel_selection import (
    test_discriminative_power,
    find_significant_channels
)
from utils.visualise import plot_discriminative_channel


parser = argparse.ArgumentParser(
    description="Find and visualise discriminative channels in ECoG data."
)


parser.add_argument(
    '--mat_file_path', required=True, type=str,
    help='Path to the .mat file containing ECoG data.'
)
parser.add_argument(
    '--figure_dir', required=False, type=str,
    help='Directory to save the figures of significant channels.'
    'Warning: This directory will be cleared before saving new figures.'
)
parser.add_argument(
    '--output_file', required=False, type=str,
    help='Path to save the output JSON file with significant channels.'
    'If the file exists, it will be updated with new data.'
)
parser.add_argument(
    '--label_names', type=str,
    nargs='+', default=['syllable', 'tone'],
    help='List of label names (columns in the mat file)'
    ' to use for the analysis.'
)
parser.add_argument(
    '--recording_name', default='ecog', type=str,
    help='Name of the recording to analyse.'
    'This should match the name used in the .mat file.'
)
parser.add_argument(
    '--p_thresholds', default=[0.01, 0.001], type=float,
    nargs='+',
    help='P-value threshold for each label.'
)
parser.add_argument(
    '--consecutive_length_thresholds', default=[50, 50], type=int,
    nargs='+',
    help='Minimum lengths of consecutive significant time points '
    'for a channel to be considered discriminative (one for each label).'
)
parser.add_argument(
    '--sampling_rate', default=400, type=int,
    help='Sampling rate of the recording. (in Hz)'
)
parser.add_argument(
    '--onset_time', default=None, type=float,
    help='Onset time of the recording in seconds.'
    'This is used to align the visualisation with the recording.'
)


args = parser.parse_args()

if len(args.p_thresholds) != len(args.label_names):
    raise ValueError(
        'Number of p-value thresholds must match the number of label names.'
    )
if len(args.consecutive_length_thresholds) != len(args.label_names):
    raise ValueError(
        'Number of consecutive length thresholds must '
        'match the number of label names.'
    )

data = loadmat(args.mat_file_path)

channel_data = {}

for i, label_name in enumerate(args.label_names):
    results = test_discriminative_power(
        data, label_name,
        recording_name=args.recording_name)

    significant_channels = find_significant_channels(
        results, pvalue_threshold=args.p_thresholds[i],
        consecutive_length_threshold=args.consecutive_length_thresholds[i]
    )
    print(f'Found {len(significant_channels)} discriminative channels '
        f'for label "{label_name}"')
    
    channel_data[f'{label_name}_discriminative'] = significant_channels


if args.output_file:
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(args.output_file):
        # append to existing
        with open(args.output_file, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
            
        existing_data.update(channel_data)

        with open(args.output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
        print('Appended discriminative channels to', args.output_file)
    else:
        with open(args.output_file, 'w') as f:
            json.dump(channel_data, f, indent=4)
        print('Saved discriminative channels to', args.output_file)
    

if args.figure_dir:
    if not os.path.exists(args.figure_dir):
        os.makedirs(args.figure_dir)
    # clear all png files that already exists
    for file in os.listdir(args.figure_dir):
        if file.endswith('.png'):
            os.remove(os.path.join(args.figure_dir, file))

    for i, significant_channels in enumerate(channel_data.values()):
        label_name = args.label_names[i]
        for ch in significant_channels:
            figure_name = '{}_channel_{}.png'.format(label_name, ch)
            figure_path = os.path.join(args.figure_dir, figure_name)

            plot_discriminative_channel(
                data, ch,
                sampling_rate=args.sampling_rate,
                label_name=label_name,
                onset_time = args.onset_time,
                recording_name='ecog', figure_path=figure_path
            )
            print(f'Channel {ch} figure saved to {figure_path}')
else:
    for label_name, significant_channels in channel_data.items():
        print(
            f'Label "{label_name}" discriminative '
            f'channels: {significant_channels}')

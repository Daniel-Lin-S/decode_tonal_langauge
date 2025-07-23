import os
import argparse
import numpy as np
import json
import pandas as pd
from scipy.io import loadmat

from data_loading.channel_selection import (
    test_discriminative_power,
    find_significant_channels
)
from utils.visualise import plot_discriminative_channel


parser = argparse.ArgumentParser(
    description="Find and visualise discriminative channels in ECoG data."
)


parser.add_argument(
    '--recording_file_path', required=True, type=str,
    help='Path to the .npz file containing ECoG data.'
)
parser.add_argument(
    '--channel_locations_file', required=True, type=str,
    help='Path to the .csv file containing channel locations.'
)
parser.add_argument(
    '--channel_locations_key', required=True, type=str,
    help='Key in the .mat file that contains the channel locations.'
)
parser.add_argument(
    '--figure_dir', required=False, type=str,
    help='Directory to save the figures of significant channels.'
    'Warning: This directory will be cleared before saving new figures.'
)
parser.add_argument(
    '--individual_figures', action='store_true',
    help='If set, save individual figures for each significant channel (electrode).'
)
parser.add_argument(
    '--output_file', required=False, type=str,
    help='Path to save the output JSON file with significant channels.'
    'If the file exists, it will be updated with new data.'
)
parser.add_argument(
    '--channel_output_file', required=False, type=str,
    help='Path to save the channel locations and discriminative scores'
    ' of all channels in a .npz file.'
    'If not given, this will not be saved.'
)
parser.add_argument(
    '--label_names', type=str,
    nargs='+', default=['syllable', 'tone'],
    help='List of label names (columns in the npz file)'
    ' to use for the analysis.'
)
parser.add_argument(
    '--recording_name', default='ecog', type=str,
    help='Name of the recording to analyse.'
    'This should match the name used in the .npz file.'
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

# -------- Value checks --------
if len(args.p_thresholds) != len(args.label_names):
    raise ValueError(
        'Number of p-value thresholds must match the number of label names.'
    )

if len(args.consecutive_length_thresholds) != len(args.label_names):
    raise ValueError(
        'Number of consecutive length thresholds must '
        'match the number of label names.'
    )

# ---------- Data Loading ----------
data = np.load(args.recording_file_path)

location_data = loadmat(args.channel_locations_file)
try:
    channel_locations = location_data[args.channel_locations_key]
except KeyError:
    raise KeyError(
        f"Key '{args.channel_locations_key}' not found in the "
        f"file '{args.channel_locations_file}'. "
        "Please check the file and the key."
        f"Available keys: {', '.join(location_data.keys())}"
    )


channel_data = {}
p_vals = {}
discriminative_powers = {}

for i, label_name in enumerate(args.label_names):
    results = test_discriminative_power(
        data, label_name,
        recording_name=args.recording_name)

    discriminative_scores = - np.log10(results['p_value'] + 1e-10)
    threshold = -np.log10(0.05)
    auc_scores = np.sum(
        np.clip(discriminative_scores - threshold, 0, None),
        axis=1
    )  # (n_electrodes,)

    significant_channels = find_significant_channels(
        results, pvalue_threshold=args.p_thresholds[i],
        consecutive_length_threshold=args.consecutive_length_thresholds[i]
    )
    print(f'Found {len(significant_channels)} discriminative channels '
        f'for label "{label_name}"')
    
    channel_data[f'{label_name}_discriminative'] = significant_channels
    p_vals[label_name] = results['p_value'][significant_channels, :]
    discriminative_powers[label_name] = auc_scores


# a dataframe to store information for all channels
if args.channel_output_file:
    df = pd.DataFrame({
        'x' : channel_locations[:, 0],
        'y' : channel_locations[:, 1],
        'z' : channel_locations[:, 2],
    })

    for label_name in args.label_names:
        df[f'{label_name}_discriminative_score'] = discriminative_powers[label_name]

    df.to_csv(args.channel_output_file, index=True)
    print(
        'Saved channel locations and discriminative scores to',
        args.channel_output_file
    )

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

    if args.individual_figures:
        for i, significant_channels in enumerate(channel_data.values()):
            label_name = args.label_names[i]

            for j, ch in enumerate(significant_channels):
                figure_name = '{}_channel_{}.png'.format(label_name, ch)
                figure_path = os.path.join(args.figure_dir, figure_name)

                plot_discriminative_channel(
                    data, ch,
                    sampling_rate=args.sampling_rate,
                    p_vals = p_vals[label_name][j, :].squeeze(),
                    label_name=label_name,
                    onset_time = args.onset_time,
                    recording_name='ecog', figure_path=figure_path
                )
                print(f'Channel {ch} figure saved to {figure_path}')

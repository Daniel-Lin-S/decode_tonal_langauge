"""
Find the channels with responses at events compared to rest period.
"""
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import json

from data_loading.channel_selection import find_active_channels
from utils.visualise import plot_rest_erp


parser = argparse.ArgumentParser(
    description="Find and visualise active channels in the recording."
)

parser.add_argument(
    '--recording_file_path', required=True, type=str,
    help='Path to the .npz file containing ECoG data.'
)
parser.add_argument(
    '--figure_dir', required=False, type=str,
    help='Directory to save the figures of significant channels.'
    'Warning: This directory will be cleared before saving new figures.'
)
parser.add_argument(
    '--output_file', required=False, type=str,
    help='Path to save the output JSON file with active channels.'
)
parser.add_argument(
    '--rest_recording_name', default='ecog_rest', type=str,
    help='Label name to use for the analysis.'
)
parser.add_argument(
    '--erp_recording_name', default='ecog', type=str,
    help='Name of the recording to analyse.'
    'This should match the name used in the .npz file.'
)
parser.add_argument(
    '--p_threshold', default=0.001, type=float,
    help='P-value threshold for significance.'
)
parser.add_argument(
    '--consecutive_length_threshold', default=50, type=int,
    help='Minimum length of consecutive significant channels.'
)
parser.add_argument(
    '--sampling_rate', default=400, type=int,
    help='Sampling rate of the ECoG data.'
    'This is used for plotting the ERP.'
)


args = parser.parse_args()

data = np.load(args.recording_file_path)

channels, lengths = find_active_channels(
    data, args.rest_recording_name,
    args.erp_recording_name, args.p_threshold,
    args.consecutive_length_threshold
)

print(f'Found {len(channels)} active channels')

if args.output_file:
    output_data = {
        'active_channels': channels
    }

    configs_dir = os.path.dirname(args.output_file)
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)
    
    if os.path.exists(args.output_file):
        # append to existing
        with open(args.output_file, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}

        existing_data.update(output_data)

        with open(args.output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
        print('Appended active channels to', args.output_file)
    else:
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print('Saved active channels to', args.output_file)


# plot the distribution of length of significant timpoints
if args.figure_dir:
    if not os.path.exists(args.figure_dir):
        os.makedirs(args.figure_dir)

    figure_path = os.path.join(args.figure_dir, 'active_channels_length.png')
    plt.figure(figsize=(10, 6))
    lengths = np.array(lengths) / args.sampling_rate  # convert to seconds
    plt.hist(lengths, bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of Active Length of Significant Channels')
    plt.xlabel('Active length (s)')
    plt.ylabel('Frequency')
    plt.savefig(figure_path, dpi=400)
    plt.close()
    print('Saved distribution of lengths of significant channels to ',
          figure_path)

    # plot 10 example channels
    n_channels_plot = min(10, len(channels))
    for ch in channels[:n_channels_plot]:
        fig_name = f'channel_{ch}_erp_rest.png'
        figure_path = os.path.join(args.figure_dir, fig_name)

        plot_rest_erp(
            data, args.rest_recording_name,
            args.erp_recording_name,
            ch,
            sampling_rate = args.sampling_rate,
            figure_path=figure_path
        )
        print('Saved ERP plot for channel', ch, 'to', figure_path)

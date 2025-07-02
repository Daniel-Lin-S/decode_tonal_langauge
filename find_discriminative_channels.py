import os
import argparse
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
    '--mat_file_path', required=True, type=str,
    help='Path to the .mat file containing ECoG data.'
)
parser.add_argument(
    '--figure_dir', required=False, type=str,
    help='Directory to save the figures of significant channels.'
    'Warning: This directory will be cleared before saving new figures.'
)
parser.add_argument(
    '--label_name', default='syllable', type=str,
    help='Label name to use for the analysis.'
)
parser.add_argument(
    '--recording_name', default='ecog', type=str,
    help='Name of the recording to analyse.'
    'This should match the name used in the .mat file.'
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
    help='Sampling rate of the recording. (in Hz)'
)
parser.add_argument(
    '--onset_time', default=None, type=float,
    help='Onset time of the recording in seconds.'
    'This is used to align the visualisation with the recording.'
)
parser.add_argument(
    '--plot', action='store_true',
    help='If set, will plot the significant channels.'
    'If not set, will only print the significant channels.'
)


args = parser.parse_args()


data = loadmat(args.mat_file_path)
results = test_discriminative_power(
    data, args.label_name,
    recording_name=args.recording_name)

# Print results
significant_channels = find_significant_channels(
    results, pvalue_threshold=args.p_threshold,
    consecutive_length_threshold=args.consecutive_length_threshold
)
print(f'Found {len(significant_channels)} significant channels.')


if args.plot:
    if not os.path.exists(args.figure_dir):
        os.makedirs(args.figure_dir)
    # clear all png files that already exists
    for file in os.listdir(args.figure_dir):
        if file.endswith('.png'):
            os.remove(os.path.join(args.figure_dir, file))

    for ch in significant_channels:
        figure_name = '{}_channel_{}.png'.format(args.label_name, ch)
        figure_path = os.path.join(args.figure_dir, figure_name)

        plot_discriminative_channel(
            args.mat_file_path, ch,
            sampling_rate=args.sampling_rate,
            label_name=args.label_name,
            onset_time = args.onset_time,
            recording_name='ecog', figure_path=figure_path
        )
        print(f'Channel {ch} figure saved to {figure_path}')
else:
    print('Significant channels: ', significant_channels)

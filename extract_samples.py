"""
Align ECoG and audio samples using TextGrid files marking the intervals
for each read word or syllable.

The name of each interval in the TextGrid should have format
[tone][syllable_identifier],
where the tone is the digit id for tone and syllable_identifier is a short string
marking the syllable spoken in the interval, e.g. "0i", "1a", etc.
 
Required hyper-parameters from the JSON file:
- rest_period: a tuple (start, end) indicating the rest period for ECoG extraction.
  will be used for referencing and active channel selection.
- syllable_identifiers: a list of syllable identifiers to be used for extracting
  syllables in the TextGrid files.

File structure of each saved mat file:
- 'ecog': The ECoG data within each interval (event), a numpy array of shape 
    (n_samples, n_channels, n_timepoints).
    where n_samples stands for the number of intervals.
- 'audio': The audio data within each interval, a numpy array of shape
    (n_samples, n_timepoints).
- 'syllable': The syllable spoken in each sample, shape (n_samples,)
- 'tone': The tone id for each sample, shape (n_samples,).
- 'ecog_rest': The ECoG data during the rest period, a numpy array of shape
    (n_rest_samples, n_channels, n_timepoints).
"""

from data_loading.text_align import handle_textgrids, extract_ecog_audio
import os
import argparse
import json


parser = argparse.ArgumentParser(
    description = "Extract ECoG and audio samples for each onset based on the textgrids")
# --------- I/O ---------
parser.add_argument(
    '--textgrid_dir', default='processed/annotation', type=str,
    help='Directory containing TextGrid files (for all blocks).'
    'Each .TextGrid file must have a naming convention that includes '
    'a block number at the end, i.e. "_B[block_number].TextGrid".'
)
parser.add_argument(
    '--recording_dir', default='processed/mat', type=str,
    help='Directory containing ECoG and audio files (for all blocks).'
    'Should have ECoG files with "ecog" in the name and '
    'audio files with "sound" in the name.'
    'Each file must start with "B[block_number]".'
)
parser.add_argument(
    '--config_file', required=True, type=str,
    help='Path to the JSON file with necessary hyperparameters.'
)
parser.add_argument(
    '--audio_kwords', default=None, type=str, nargs='+',
    help='List of keywords to identify audio files. '
    'only files containing these keywords will be considered for audio extraction. '
    'Defaults to None.'
)
parser.add_argument(
    '--ecog_kwords', default=None, type=str, nargs='+',
    help='List of keywords to identify ECoG files. '
    'only files containing these keywords will be considered for ECoG extraction. '
    'Defaults to None.'
)
parser.add_argument(
    '--output_path', default='data/samples/samples.mat', type=str,
    help='Path to save the output .mat file containing the extracted samples.'
)
parser.add_argument(
    '--blocks', nargs='+', type=int, default=None,
    help='List of block numbers to process. If None, all blocks will be processed.'
)


# TODO move this into a config file
syllables = ['i', 'a']  # syllable marks in the TextGrid files.
rest_period = (0.0, 25.0)  # Default rest period for ECoG extraction.


args = parser.parse_args()

if __name__ == '__main__':
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    rest_period = config['rest_period']
    if len(rest_period) != 2 or not all(
        isinstance(x, (int, float)) for x in rest_period):
        raise ValueError(
            "rest_period must be a list of two floats (start, end). "
            f"Received: {rest_period}"
        )

    syllable_identifiers = config['syllable_identifiers']
    rest_period = tuple(config['rest_period'])

    if os.path.exists(args.output_path):
        print(f"Output file {args.output_path} already exists. "
              "Skipping ...")
        exit(1)

    print('----------- '
        f'Extracting all samples from {args.textgrid_dir} and {args.recording_dir}'
        ' -----------')

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    intervals = handle_textgrids(
        args.textgrid_dir, start_offset=0.2, tier_list=['success'],
        blocks=args.blocks)
    if len(intervals) == 0:
        raise ValueError(
            "No intervals found in the TextGrid files. "
            "Check the directory and file naming conventions."
            f"Target blocks: {args.blocks if args.blocks else 'all'}"
        )

    print(f"Extracted intervals from TextGrid files: {len(intervals)} blocks found.")
    block_numbers = list(intervals.keys())

    extract_ecog_audio(
        intervals, args.recording_dir,
        syllables,
        audio_kwords=args.audio_kwords,
        ecog_kwords=args.ecog_kwords,
        output_path=args.output_path,
        rest_period=rest_period
    )

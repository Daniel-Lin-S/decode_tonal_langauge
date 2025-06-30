from data_loading.text_align import handle_textgrids, extract_ecog_audio
import os
import argparse


parser = argparse.ArgumentParser(
    description = "Extract ECoG and audio samples for each onset based on the textgrids")
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
    '--audio_kwords', default=None, type=str, nargs='+',
    help='List of keywords to identify audio files. '
         'Defaults to None.'
)
parser.add_argument(
    '--ecog_kwords', default=None, type=str, nargs='+',
    help='List of keywords to identify ECoG files. '
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

syllables = ['i', 'a']


args = parser.parse_args()

if __name__ == '__main__':
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
    )

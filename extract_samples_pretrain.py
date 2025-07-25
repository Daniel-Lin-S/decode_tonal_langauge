import argparse
import os
import numpy as np

from data_loading.dataloaders import collect_unlabelled_samples


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Extract unlabelled samples from ECoG dataset for pre-training'
    )

    # -------- I/O Parameters --------
    parser.add_argument(
        '--dataset_dir', type=str, required=True,
        help='Path to the dataset directory containing ECoG data'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/pretrain',
        help='Directory to save the extracted samples.'
        ' (default: data/pretrain)'
    )
    parser.add_argument(
        '--patch_size', type=int, default=200,
        help='Size of the input patches in time points (default: 200)'
    )
    parser.add_argument(
        '--segment_time', type=int, default=5,
        help='Length of each segment in seconds (default: 30)'
    )
    parser.add_argument(
        '--sampling_rate', type=int, default=200,
        help='Sampling rate of the data in Hz (default: 200)'
    )
    parser.add_argument(
        '--ecog_kwords', type=str, nargs='+',
        default=['ecog'],
        help='Keywords to filter ECoG data files (default: "ecog")'
    )

    return parser.parse_args()


if __name__ == '__main__':
    params = get_arguments()

    if not os.path.exists(params.dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory '{params.dataset_dir}' does not exist."
        )

    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    samples = collect_unlabelled_samples(
        dataset_folder=params.dataset_dir,
        patch_size=params.patch_size,
        segment_length=params.segment_time * params.sampling_rate,
        kwords=params.ecog_kwords
    )

    print('Shape of extracted samples:', samples.shape)

    file_name = 'ecog_patches_{}_{}.npy'.format(
        params.segment_time, params.patch_size
    )

    output_path = f"{params.output_dir}/{file_name}"
    
    np.save(output_path, samples)
    print('Saved extracted samples to:', output_path)

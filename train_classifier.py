"""
Data file should have keys "ecog" with shape
(n_samples, n_channels, n_timepoints)
and "{target}" with shape (n_samples,)
where target is the classification target, e.g. "syllable" or "tone".
Each sample corresponds to a trial in the experiment.
Training is implemented with PyTorch Lightning for clearer logging.

Required hyper-parameters from the JSON file:
- syllable_labels: a list of syllable labels, used for visualising the classification results.
  For example, ["ba", "da"] will map label 0 to "ba" and label 1 to "da".
  If not provided, the labels will be numbered from 1 to n_classes.
- tone_labels: a list of tone labels, used for visualising the classification results.
  If not provided, the labels will be numbered from 1 to n_classes.
- classifier_kwargs: a dictionary of keyword arguments for the classifier.
  If not given, the default values will be used.
"""

import argparse
import torch
import os
import numpy as np
import json

from training.classifier_pipeline import (
    train_joint_targets, train_separate_targets, save_and_plot_results
)
from models.classifier_factory import MODEL_CHOICES


def parse_train_classifier_args():
    parser = argparse.ArgumentParser(
        description="Train a classifier on ECoG data."
    )

    # ----- I/O -------
    parser.add_argument(
        '--sample_path', type=str, required=True,
        help='The npz file containing the ECoG samples. ')
    parser.add_argument(
        '--figure_dir', type=str, default='figures',
        help='Directory to save the figures.'
    )
    parser.add_argument(
        '--channel_file', type=str, default=None,
        help='JSON file containing channel selections for the model. '
        'Must have "syllable_discriminative" and "tone_discriminative" keys.'
    )
    parser.add_argument(
        '--config_file', required=True, type=str,
        help='Path to the JSON file with necessary hyperparameters'
    )
    parser.add_argument(
        '--result_file', required=True, type=str,
        help='Path to the csv file to save the results. '
    )
    parser.add_argument(
        '--model_dir', type=str, required=False,
        default=None,
        help='Directory to save the trained model. '
        'If not specified, the model will not be saved.'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs',
        help='Directory to save the pytorch lightning training logs. '
        'Including a tensorboard and a csv logger.'
    )
    parser.add_argument(
        '--subject_id', type=int, required=True,
        help='Subject ID, used to name the output files.'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        choices=MODEL_CHOICES,
        help='The classification model to be used.'
    )
    parser.add_argument(
        '--model_name', type=str, required=True,
        help='Name of the model to be trained. Will be used to name the output files.'
    )
    parser.add_argument(
        '--foundation_weights_path', type=str, default=None,
        help='Path to the pre-trained weights for the foundation model. '
    )
    # ----- Experiment Settings -------
    parser.add_argument(
        '--targets', type=str, required=True, nargs='+',
        choices=["syllable", "tone"],
        help='The target variable to classify. '
    )
    parser.add_argument(
        '--separate_models', action='store_true',
        help='If set, train separate models for each target. ' \
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility. Default is 42.'
    )
    parser.add_argument(
        '--repeat', type=int, default=1,
        help='Number of times to repeat the training. Default is 1.'
    )
    parser.add_argument(
        '--verbose', type=int, default=1,
        help='Verbosity level of the training process. ' \
        '0: Only the final accuracies, 1: Basic output each run (repeat), '
        '2: Detailed output for each epoch.'
    )
    # ----- Training settings -------
    parser.add_argument(
        '--train_ratio', type=float, default=0.7,
        help='Ratio of the dataset to use for training. Default is 0.7.')
    parser.add_argument(
        '--vali_ratio', type=float, default=0.1,
        help='Ratio of the dataset to use for validation. Default is 0.1.')
    parser.add_argument(
        '--test_ratio', type=float, default=0.2,
        help='Ratio of the dataset to use for testing. Default is 0.2.')
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device to use for training. Default is "cuda". ' \
        'Use "cpu" for CPU training.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size for training. Default is 64.')
    parser.add_argument(
        '--epochs', type=int, default=25,
        help='Number of epochs to train the model. Default is 25.')
    parser.add_argument(
        '--lr', type=float, default=0.0005,
        help='Learning rate for the optimizer. Default is 0.0005.')
    parser.add_argument(
        '--patience', type=int, default=5,
        help='Number of epochs with no improvement after which '
        'training will be stopped. Default is 5.'
    )
    parser.add_argument(
        '--log_every_n_steps', type=int, default=10,
        help='Log training progress every n steps. Default is 10.'
    )

    args = parser.parse_args()

    # ------ Value checks -------
    if 'cuda' in args.device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please use 'cpu' as device.")

    if args.train_ratio + args.vali_ratio + args.test_ratio != 1.0:
        raise ValueError(
            "The sum of train_ratio, vali_ratio, and test_ratio must be 1.0. "
            f"Current values: train_ratio={args.train_ratio}, "
            f"vali_ratio={args.vali_ratio}, test_ratio={args.test_ratio}")

    if not os.path.exists(args.sample_path):
        raise FileNotFoundError(
            f"Data file '{args.sample_path}' does not exist.")

    # ------- Create directories if they do not exist -------
    if not os.path.exists(args.figure_dir):
        os.makedirs(args.figure_dir)

    if args.model_dir is not None and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    results_dir = os.path.dirname(args.result_file)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    return args


if __name__ == '__main__':
    params = parse_train_classifier_args()

    with open(params.config_file, 'r') as f:
        config = json.load(f)
    
    syllable_labels = config.get('syllable_labels', None)
    tone_labels = config.get('tone_labels', None)
    classifier_kwargs = config.get(f'classifier_kwargs', {})

    np.random.seed(params.seed)
    seeds = np.random.randint(0, 10000, params.repeat)

    if params.separate_models:
        results, conf_mat, labels = train_separate_targets(
            params, config, seeds
        )

    else:
        results, conf_mat, labels = train_joint_targets(
            params, config, seeds
        )

    save_and_plot_results(
        params, results, conf_mat, labels
    )

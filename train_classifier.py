"""
Data file should have keys "ECoG_toneT{tone}" with shape
(n_samples, n_channels, n_timepoints)
where {tone} is the index for tone.
Each sample corresponds to a trial in the experiment. 
"""

import argparse
from scipy.io import loadmat
import torch
from torch.utils.data import TensorDataset
import os
import numpy as np

from models import syllableModel, toneModel
from models.classifierTrainer import ClassifierTrainer
from utils.utils import set_seeds
from utils.visualise import plot_training_losses, plot_confusion_matrix

from data_loading.dataloaders import split_dataset


parser = argparse.ArgumentParser(
    description="Train a classifier on ECoG data.")

# ----- I/O -------
parser.add_argument(
    '--sample_path', type=str, required=True,
    help='The mat file containing the ECoG samples. ')
parser.add_argument(
    '--figure_dir', type=str, default='figures',
    help='Directory to save the figures.'
)
# ----- Experiment Settings -------
parser.add_argument(
    '--target', type=str, required=True,
    help='The target variable to classify. '
    'Options: ["syllables", "tones"]'
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
    '--train_ratio', type=float, default=0.9,
    help='Ratio of the dataset to use for training. Default is 0.9.')
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


# TODO make these into a config file
syllables_discriminative_channels = [
    70, 71, 86, 180, 182, 195, 196, 197, 198, 211, 212, 227, 228]
tone_discriminative_channels = [
    9, 10, 105, 120, 121, 184, 200, 206]

syllable_labels = ['mi', 'ma']

if __name__ == '__main__':
    params = parser.parse_args()

    # check CUDA availability
    if 'cuda' in params.device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please use 'cpu' as device.")

    if not os.path.exists(params.figure_dir):
        os.makedirs(params.figure_dir)

    np.random.seed(params.seed)
    seeds = np.random.randint(0, 10000, params.repeat)

    if not os.path.exists(params.sample_path):
        raise FileNotFoundError(
            f"Data file '{params.sample_path}' does not exist.")

    dataset = loadmat(params.sample_path)

    # TODO add channel selector
    if params.target == 'tones':
        all_erps, labels = dataset['ecog'], dataset['tone'].flatten()
        all_erps = all_erps[:, tone_discriminative_channels, :]
    elif params.target == 'syllables':
        all_erps, labels = dataset['ecog'], dataset['syllable'].flatten()
        all_erps = all_erps[:, syllables_discriminative_channels, :]
    else:
        raise ValueError(
            f"Invalid target '{params.target}'. "
            "Choose either 'tones' or 'syllables'.")

    erps_tensor = torch.tensor(
        all_erps, dtype=torch.float32).to(params.device)
    labels_tensor = torch.tensor(
        labels, dtype=torch.float32).to(params.device)

    dataset_tensor = TensorDataset(erps_tensor, labels_tensor)
    n_samples, n_channels, seq_length = erps_tensor.shape

    if params.verbose > 0:
        print(f"Prepared {n_samples} samples with shape {erps_tensor.shape}"
              f" and labels with shape {labels_tensor.shape}")

    n_classes = len(np.unique(labels))

    accuracies = []
    f1_scores = []
    losses = []
    confusion_mat = np.zeros((n_classes, n_classes))

    for i, seed in enumerate(seeds):
        set_seeds(seed)

        train_loader, test_loader = split_dataset(
            dataset_tensor, params.train_ratio, params.batch_size,
            seed=seed
        )

        trainer_verbose = params.verbose > 1

        if params.target == 'syllables':
            model = syllableModel.Model(
                n_channels, seq_length, n_classes).to(params.device)
        elif params.target == 'tones':
            model = toneModel.Model(
                n_channels, seq_length, n_classes).to(params.device)

        if params.verbose > 0 and i == 0:
            print(f"Model summary: {model.get_layer_nparams()}")

        trainer = ClassifierTrainer(
            model, device=params.device,
            learning_rate=params.lr,
            verbose=False
        )

        if params.verbose > 0:
            print(f"Training with seed {seed} ...")

        history = trainer.train(
            train_loader, epochs=params.epochs,
            verbose=trainer_verbose
        )

        accuracy, f1, conf = trainer.evaluate(test_loader)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        confusion_mat += conf

        if params.verbose > 0:
            print(f"Run {i+1} / {params.repeat}: "
                  f"Test accuracy: {accuracy:.4f}, "
                  f"F1 score: {f1:.4f}")

        losses.append([loss for loss, _ in history])

    print(f"-------- Training completed over {params.repeat} runs --------")
    print(
        f"Average accuracy over {params.repeat} runs: {np.mean(accuracies):.4f}"
        f" ± {np.std(accuracies):.4f}")
    print(
        f"Average F1 score over {params.repeat} runs: {np.mean(f1_scores):.4f}"
        f" ± {np.std(f1_scores):.4f}"
    )

    plot_training_losses(
        losses,
        figure_path=os.path.join(params.figure_dir, 'training_losses.png')
    )

    # only add numbers to the confusion matrix plot
    # if classes are few to avoid visual clutter
    add_numbers = n_classes <= 10
    if params.target == 'tones':
        class_labels = None
    elif params.target == 'syllables':
        class_labels = syllable_labels

    plot_confusion_matrix(
        confusion_mat, add_numbers, label_names=class_labels,
        figure_path=os.path.join(params.figure_dir, 'confusion_matrix.png')
    )

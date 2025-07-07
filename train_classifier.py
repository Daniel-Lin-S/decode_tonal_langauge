"""
Data file should have keys "ECoG_toneT{tone}" with shape
(n_samples, n_channels, n_timepoints)
where {tone} is the index for tone.
Each sample corresponds to a trial in the experiment.

Required hyper-parameters from the JSON file:
- syllable_labels: a list of syllable labels, used for visualising the classification results.
  For example, ["ba", "da"] will map label 0 to "ba" and label 1 to "da".
"""

import argparse
import torch
from torch.utils.data import TensorDataset
import os
import numpy as np
import json
import pandas as pd

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
    help='The npz file containing the ECoG samples. ')
parser.add_argument(
    '--figure_dir', type=str, default='figures',
    help='Directory to save the figures.'
)
parser.add_argument(
    '--channel_file', type=str, default='channel_selections.json',
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
    '--subject_id', type=int, required=True,
    help='Subject ID, used to name the output files.'
)
parser.add_argument(
    '--model_name', type=str, required=True,
    help='Name of the model to be trained. Will be used to name the output files.'
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


if __name__ == '__main__':
    params = parser.parse_args()

    # ------ Value checks -------
    if 'cuda' in params.device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please use 'cpu' as device.")

    if params.train_ratio + params.vali_ratio + params.test_ratio != 1.0:
        raise ValueError(
            "The sum of train_ratio, vali_ratio, and test_ratio must be 1.0. "
            f"Current values: train_ratio={params.train_ratio}, "
            f"vali_ratio={params.vali_ratio}, test_ratio={params.test_ratio}")
    
    if params.target not in ['tones', 'syllables']:
        raise ValueError(
            f"Invalid target '{params.target}'. "
            "Choose either 'tones' or 'syllables'.")
    
    if not os.path.exists(params.sample_path):
        raise FileNotFoundError(
            f"Data file '{params.sample_path}' does not exist.")

     # ------- Load configuration file -------
    with open(params.config_file, 'r') as f:
        config = json.load(f)
    
    syllable_labels = config.get('syllable_labels')

    # ------- Create directories if they do not exist -------
    if not os.path.exists(params.figure_dir):
        os.makedirs(params.figure_dir)

    if params.model_dir is not None and not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)

    results_dir = os.path.dirname(params.result_file)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # ------- Prepare dataset -------

    dataset = np.load(params.sample_path)

    if params.target == 'tones':
        with open(params.channel_file, 'r') as f:
            channel_selections = json.load(f)
        tone_discriminative_channels = channel_selections[
            'tone_discriminative']

        all_erps, labels = dataset['ecog'], dataset['tone'].flatten()
        all_erps = all_erps[:, tone_discriminative_channels, :]
    elif params.target == 'syllables':
        with open(params.channel_file, 'r') as f:
            channel_selections = json.load(f)
        syllable_discriminative_channels = channel_selections[
            'syllable_discriminative']

        all_erps, labels = dataset['ecog'], dataset['syllable'].flatten()
        all_erps = all_erps[:, syllable_discriminative_channels, :]
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

    # ------- Start experiments -------
    np.random.seed(params.seed)
    seeds = np.random.randint(0, 10000, params.repeat)

    accuracies = []
    f1_scores = []
    train_losses = []
    vali_losses = []
    confusion_mat = np.zeros((n_classes, n_classes))

    for i, seed in enumerate(seeds):
        set_seeds(seed)

        data_loaders = split_dataset(
            dataset_tensor, 
            [params.train_ratio, params.vali_ratio, params.test_ratio],
            shuffling=[True, False, False],
            batch_size=params.batch_size,
            seed=seed
        )

        trainer_verbose = params.verbose > 1

        if params.target == 'syllables':
            model = syllableModel.Model(
                n_channels, seq_length, n_classes).to(params.device)
        elif params.target == 'tones':
            model = toneModel.Model(
                n_channels, seq_length, n_classes).to(params.device)

        model_verbose = params.verbose > 0 and i == 0
        if model_verbose:
            print(f"Number of trainable parameters: {model.get_layer_nparams()}")

        trainer = ClassifierTrainer(
            model, device=params.device,
            learning_rate=params.lr,
            verbose=model_verbose
        )

        if params.verbose > 0:
            print(f"Training with seed {seed} ...")

        history = trainer.train(
            data_loaders[0], epochs=params.epochs,
            vali_loader=data_loaders[1],
            patience=params.patience,
            verbose=trainer_verbose
        )

        # save the model
        if params.model_dir is not None:
            model_save_path = os.path.join(
                params.model_dir, f"{params.target}_model_seed_{seed}.pt")
            
            torch.save(model.state_dict(), model_save_path)
            
            if params.verbose > 0:
                print(f"Model saved to {model_save_path}")

        accuracy, f1, conf = trainer.evaluate(data_loaders[2])
        accuracies.append(accuracy)
        f1_scores.append(f1)
        confusion_mat += conf

        if params.verbose > 0:
            print(f"Run {i+1} / {params.repeat}: "
                  f"Test accuracy: {accuracy:.4f}, "
                  f"F1 score: {f1:.4f}")

        train_losses.append([item['train_loss'] for item in history])
        vali_losses.append([item['vali_loss'] for item in history])


    # --------- Save results ---------
    print(f"-------- Training completed over {params.repeat} runs --------")
    print(
        f"Average accuracy over {params.repeat} runs: {np.mean(accuracies):.4f}"
        f" ± {np.std(accuracies):.4f}")
    print(
        f"Average F1 score over {params.repeat} runs: {np.mean(f1_scores):.4f}"
        f" ± {np.std(f1_scores):.4f}"
    )

    experiment_results = {
        'model_name': params.model_name,
        'subject': params.subject_id,
        'target' : params.target,
        'seeds' : str(seeds),
        'learning_rate' : params.lr,
        'epochs': params.epochs,
        'patience': params.patience,
        'batch_size': params.batch_size,
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'all_accuracies': str(accuracies),
        'all_f1_scores': str(f1_scores),
        'confusion_matrix': str(confusion_mat.tolist())
    }

    results_df = pd.DataFrame([experiment_results])

    if os.path.exists(params.result_file):
        results_df.to_csv(
            params.result_file, mode='a', header=False, index=False
        )
    else:
        results_df.to_csv(
            params.result_file, index=False
        )
    print(f"Results saved to {params.result_file}")

    plot_training_losses(
        train_losses, vali_losses=vali_losses,
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

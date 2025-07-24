"""
Data file should have keys "ecog" with shape
(n_samples, n_channels, n_timepoints)
and "{target}" with shape (n_samples,)
where target is the classification target, e.g. "syllable" or "tone".
Each sample corresponds to a trial in the experiment.

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
from torch.utils.data import TensorDataset
import os
import numpy as np
import json
import pandas as pd
from typing import Tuple, Dict, List, Optional

from models.classifier_trainer import ClassifierTrainer
from models.simple_classifiers import LogisticRegressionClassifier, ShallowNNClassifier
from models.deep_classifiers import CNNClassifier, CNNRNNClassifier
from models.classifier import ClassifierModel
from models.cbramod_classifier import CBraModClassifier
from utils.utils import set_seeds, prepare_class_labels
from utils.visualise import plot_training_losses, plot_confusion_matrix
from utils.metrics import compute_joint_metrics

from data_loading.dataloaders import split_dataset


parser = argparse.ArgumentParser(
    description="Train a classifier on ECoG data.")


model_choices = ['logistic', 'CNN', 'CNN-RNN', 'ShallowNN', 'CBraMod']

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
    '--subject_id', type=int, required=True,
    help='Subject ID, used to name the output files.'
)
parser.add_argument(
    '--model', type=str, required=True,
    choices=model_choices,
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


def prepare_erps(
    sample_path: str,
    targets: List[str],
    channel_file: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load the ECoG dataset and prepare the ERPs and labels for training.

    Parameters
    ----------
    sample_path : str
        Path to the npz file containing the samples.
    targets : List[str]
        List of target variables to classify, e.g. ["syllable", "tone"].
    channel_file : Optional[str], default=None
        Path to the JSON file containing channel selections for the model.
        If None, all channels will be used.

    Returns
    -------
    all_erps : np.ndarray
        The ECoG data with shape (n_samples, n_channels, n_timepoints).
    labels : np.ndarray
        The labels corresponding to the ECoG data with shape (n_samples,).
    channels : np.ndarray
        The indices of the channels used for training, based on the channel file.
    n_classes_dict : Dict[str, int]
        A dictionary mapping each target to the
        number of unique classes in that target.
    """
    dataset = np.load(sample_path)

    try:
        all_erps = dataset['ecog']
    except KeyError:
        raise KeyError(
            "The dataset does not contain 'ecog' key. "
            "Please check the data file. "
            f"Available keys in the file: {', '.join(dataset.keys())}"
        )

    target_labels = []
    n_classes_dict = {}
    for target in targets:
        if target not in dataset:
            raise KeyError(
                f"The dataset does not contain '{target}' key. "
                "Please check the data file. "
                f"Available keys in the file: {', '.join(dataset.keys())}"
            )

        target_labels.append(dataset[target].flatten())
        n_classes_dict[target] = len(np.unique(dataset[target]))

    # combine target labels into a single label array
    labels = np.zeros_like(target_labels[0], dtype=int)
    multiplier = 1
    for target_label in target_labels:
        labels += target_label * multiplier
        multiplier *= len(np.unique(target_label))

    # ------ filter channels -------
    if channel_file is not None:
        with open(channel_file, 'r') as f:
            channel_selections = json.load(f)

        channels = set()

        # Loop through all targets and union their discriminative channels
        for target in targets:
            if f'{target}_discriminative' not in channel_selections:
                raise KeyError(
                    f"Channel selection for '{target}_discriminative' "
                    f"not found in the file {channel_file}. \n"
                    "Please check the channel_file or the target variable. "
                    f"Available keys in the file: {', '.join(channel_selections.keys())}"
                )
            channels.update(channel_selections[f'{target}_discriminative'])

        # Convert the set back to a sorted list
        channels = sorted(channels)

        if len(channels) == 0:
            raise ValueError(
                f"No channels found for the targets: {', '.join(targets)}. "
                "Please check the channel file."
            )
    else:
        channels = np.arange(0, all_erps.shape[1])

    all_erps = all_erps[:, channels, :]

    return all_erps, labels, channels, n_classes_dict


def build_classifier(
        params: argparse.Namespace,
        n_classes: int,
        n_channels: int,
        seq_length: int,
        classifier_kwargs: dict={}
    ) -> ClassifierModel:
    """
    Build the classifier model based on the specified parameters.

    Parameters
    ----------
    params : argparse.Namespace
        The parameters parsed from the command line.
    n_classes : int
        The number of classes for classification.
    n_channels : int
        The number of channels in the input data.
    seq_length : int
        The length of the input sequence (number of timepoints).
    classifier_kwargs : dict, optional
        Additional keyword arguments for the classifier model.
        Default is an empty dictionary.
    
    Return
    ------
    ClassifierModel
        The classifier model (nn.Module) built.
    """

    if params.model == 'logistic':
        model = LogisticRegressionClassifier(
                    input_dim=n_channels * seq_length,
                    n_classes=n_classes,
                    **classifier_kwargs
                ).to(params.device)
    elif params.model == 'CNN':
        model = CNNClassifier(
                    input_channels=n_channels,
                    input_length=seq_length,
                    n_classes=n_classes,
                    **classifier_kwargs
                ).to(params.device)
    elif params.model == 'ShallowNN':
        model = ShallowNNClassifier(
                    input_dim=n_channels * seq_length,
                    n_classes=n_classes,
                    **classifier_kwargs
                ).to(params.device)
    elif params.model == 'CNN-RNN':
        model = CNNRNNClassifier(
                    input_channels=n_channels,
                    input_length=seq_length,
                    n_classes=n_classes,
                    **classifier_kwargs
                ).to(params.device)
    elif params.model == 'CBraMod':
        model = CBraModClassifier(
                    input_channels=n_channels,
                    input_length=seq_length,
                    n_classes=n_classes,
                    pretrained_weights_path=params.foundation_weights_path,
                    device=params.device,
                    **classifier_kwargs
                ).to(params.device)
    else:
        raise ValueError(
                    f"Invalid model name '{params.model}'. "
                    f"Choose from {model_choices}."
                )
        
    return model


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
    
    if not os.path.exists(params.sample_path):
        raise FileNotFoundError(
            f"Data file '{params.sample_path}' does not exist.")

    # ------- Load configuration file -------
    with open(params.config_file, 'r') as f:
        config = json.load(f)
    
    syllable_labels = config.get('syllable_labels', None)
    tone_labels = config.get('tone_labels', None)
    classifier_kwargs = config.get(f'classifier_kwargs', {})

    # ------- Create directories if they do not exist -------
    if not os.path.exists(params.figure_dir):
        os.makedirs(params.figure_dir)

    if params.model_dir is not None and not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)

    results_dir = os.path.dirname(params.result_file)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    np.random.seed(params.seed)
    seeds = np.random.randint(0, 10000, params.repeat)

    if params.separate_models:
        # train separate models for each target
        # and compute the overall accuracy

        class_labels = prepare_class_labels(
            params.targets,
            class_label_dict = {
                'syllable' : syllable_labels,
                'tone' : tone_labels
            }
        )

        all_datasets = {}
        input_shapes = {}
        channels = set()
        n_classes_dict = {}

        # ------- Prepare dataset -------
        for target in params.targets:
            all_erps, labels, channels_target, n_classes_target = prepare_erps(
                params.sample_path, [target], params.channel_file
            )
            n_classes_dict[target] = n_classes_target[target]

            channels.update(channels_target)

            erps_tensor = torch.tensor(
                all_erps, dtype=torch.float32).to(params.device)
            labels_tensor = torch.tensor(
                labels, dtype=torch.float32).to(params.device)

            dataset_tensor = TensorDataset(erps_tensor, labels_tensor)

            all_datasets[target] = dataset_tensor        

            input_shapes[target] = erps_tensor.shape[1:]

            if params.verbose > 0:
                print(f"Prepared {input_shapes[target][1]} samples with shape {erps_tensor.shape}"
                    f" and {target} labels with shape {labels_tensor.shape}")

        n_classes = 1
        for target, n_classes_target in n_classes_dict.items():
            n_classes *= n_classes_target

        # ------- Start experiments -------
        accuracies = []
        f1_scores = []
        confusion_mat = np.zeros((n_classes, n_classes))
        all_train_losses = {}
        all_vali_losses = {}
        all_confusion_mats = {}

        for i, seed in enumerate(seeds):
            set_seeds(seed)

            all_preds = {}   # predicted labels
            all_true = {}    # true labels

            model_size = 0
            for target, dataset_tensor in all_datasets.items():
                if params.verbose > 0:
                    print(f"Training for target: {target} with seed {seed}...")
                data_loaders = split_dataset(
                    dataset_tensor, 
                    [params.train_ratio, params.vali_ratio, params.test_ratio],
                    shuffling=[True, False, False],
                    batch_size=params.batch_size,
                    seed=seed
                )

                # extract true labels
                all_true[target] = []
                for batch in data_loaders[2]:
                    _, labels = batch
                    all_true[target].append(labels.cpu().numpy())

                # Concatenate all true labels into a single array
                all_true[target] = np.concatenate(all_true[target])

                trainer_verbose = params.verbose > 1

                model = build_classifier(
                    params, n_classes_dict[target],
                    input_shapes[target][0], input_shapes[target][1],
                    classifier_kwargs=classifier_kwargs
                )
                model_size += model.get_nparams()

                model_verbose = params.verbose > 0 and i == 0
                if model_verbose:
                    print(
                        f"Number of trainable parameters: {model.get_layer_nparams()}"
                    )

                trainer = ClassifierTrainer(
                    model, device=params.device,
                    learning_rate=params.lr,
                    verbose=model_verbose
                )

                if params.verbose > 0:
                    print(f"Training {target} model with seed {seed} ...")

                history = trainer.train(
                    data_loaders[0], epochs=params.epochs,
                    vali_loader=data_loaders[1],
                    patience=params.patience,
                    verbose=trainer_verbose
                )

                # save the model
                if params.model_dir is not None:
                    model_save_path = os.path.join(
                        params.model_dir, f"{target}_{params.model_name}_seed_{seed}.pt"
                    )
                    
                    torch.save(model.state_dict(), model_save_path)
                    
                    if params.verbose > 0:
                        print(f"Model saved to {model_save_path}")

                metrics, preds = trainer.evaluate(data_loaders[2], return_preds=True)
                all_preds[target] = preds.cpu().numpy()

                if target in all_train_losses:
                    all_train_losses[target].append(
                        [item['train_loss'] for item in history]
                    )
                    all_vali_losses[target].append(
                        [item['vali_loss'] for item in history]
                    )
                    all_confusion_mats[target] += metrics['confusion_matrix']
                else:
                    all_train_losses[target] = [
                        [item['train_loss'] for item in history]
                    ]
                    all_vali_losses[target] = [
                        [item['vali_loss'] for item in history]
                    ]
                    all_confusion_mats[target] = metrics['confusion_matrix']

            metrics = compute_joint_metrics(
                all_true, all_preds,
                metrics = ['accuracy', 'f1_score', 'confusion_matrix']
            )
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_score'])
            confusion_mat += metrics['confusion_matrix']

    else:  # build a joint dataset for all targets
        # ------- Prepare dataset -------

        all_erps, labels, channels, n_classes_dict = prepare_erps(
            params.sample_path, params.targets, params.channel_file
        )

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

        class_labels = prepare_class_labels(
            params.targets,
            n_classes_dict=n_classes_dict,
            class_label_dict = {
                'syllable' : syllable_labels,
                'tone' : tone_labels
            }
        )

        print('Number of classes ', n_classes)

        # ------- Start experiments -------
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

            model = build_classifier(
                model_choices, params, classifier_kwargs,
                n_classes, n_channels, seq_length
            )
            model_size = model.get_nparams()

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
                target_str = '_'.join(params.targets) if len(params.targets) > 1 else params.targets[0]
                model_save_path = os.path.join(
                    params.model_dir, f"{target_str}_{params.model_name}_seed_{seed}.pt"
                )
                
                torch.save(model.state_dict(), model_save_path)
                
                if params.verbose > 0:
                    print(f"Model saved to {model_save_path}")

            metrics = trainer.evaluate(data_loaders[2])
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_score'])
            confusion_mat += metrics['confusion_matrix']

            if params.verbose > 0:
                print(f"Run {i+1} / {params.repeat}: "
                    f"Test accuracy: {metrics['accuracy']:.4f}, "
                    f"F1 score: {metrics['f1_score']:.4f}")

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
        'model_kwargs' : str(classifier_kwargs),
        'model_size': model_size,
        'subject': params.subject_id,
        'target' : ','.join(params.targets),
        'electrodes': str(channels),
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
        'all_f1_scores': str(f1_scores)
    }

    if not params.separate_models:
        experiment_results['confusion_matrix'] = str(confusion_mat.tolist())

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

    if params.separate_models:
        for target in params.targets:
            plot_training_losses(
                all_train_losses[target], vali_losses=all_vali_losses[target],
                figure_path=os.path.join(
                    params.figure_dir, f'training_losses_{target}.png'
                )
            )

    else:
        plot_training_losses(
            train_losses, vali_losses=vali_losses,
            figure_path=os.path.join(params.figure_dir, 'training_losses.png')
        )

    # only add numbers to the confusion matrix plot
    # if classes are few to avoid visual clutter
    add_numbers = n_classes <= 10

    plot_confusion_matrix(
        confusion_mat, add_numbers, label_names=class_labels,
        figure_path=os.path.join(params.figure_dir, 'confusion_matrix.png')
    )

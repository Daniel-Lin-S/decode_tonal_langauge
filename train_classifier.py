"""Train a classifier on ECoG data using a YAML configuration."""

import os
import numpy as np
import yaml
from argparse import Namespace

from training.classifier_pipeline import (
    train_joint_targets,
    train_separate_targets,
    save_and_plot_results,
)
from utils.config import (
    load_config, dict_to_namespace,
    generate_hash_name_from_config
)


def run(config: dict) -> str:
    """Run classifier training from a configuration dictionary."""
    print('Running train_classifier ...')

    training_section = config.get("training", {})
    train_cfg = training_section.get("params", {})
    flat_train = {}
    for section in ("io", "experiment", "training"):
        flat_train.update(train_cfg.get(section, {}))
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    evaluation_cfg = config.get("evaluation", {})

    combined_cfg = {**flat_train, **dataset_cfg, **model_cfg, **evaluation_cfg}

    params = dict_to_namespace(combined_cfg, exclude_keys=['class_labels', 'model_kwargs'])

    sample_dir = getattr(params, "sample_dir", "data/samples")
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(
            f"Sample directory {sample_dir} does not exist."
            "Please specify a valid sample_dir in the config."
        )
    
    subject_files = [
        f for f in os.listdir(sample_dir)
        if f.endswith('.npz') and f.startswith('subject_')
    ]

    if not subject_files:
        raise FileNotFoundError(
            f"No subject files found in {sample_dir}. "
            "Ensure files are named like 'subject_<id>.npz'."
        )

    # Set default model name if not provided
    if getattr(params, "model_name", None) is None and "model" in model_cfg:
        params.model_name = model_cfg["model"].split(".")[-1]

    base_log_dir = getattr(params, "log_dir", "logs")
    exp_name = generate_hash_name_from_config(
        getattr(params, 'model_name', 'model'),
        config=combined_cfg,
    )

    params.log_dir = os.path.join(base_log_dir, exp_name)

    # Copy config to log directory for record keeping
    os.makedirs(params.log_dir, exist_ok=True)
    sample_cfg_path = os.path.join(params.sample_dir, 'config.yaml')
    if os.path.exists(sample_cfg_path):
        base_cfg = load_config(sample_cfg_path)
    else:
        base_cfg = {}

    merged_cfg = {
        **base_cfg,
        'model': model_cfg,
        'training': training_section,
        'dataset': dataset_cfg,
        'evaluation': evaluation_cfg,
    }

    with open(os.path.join(params.log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(merged_cfg, f)

    np.random.seed(getattr(params, "seed", 42))
    seeds = np.random.randint(0, 10000, getattr(params, "repeat", 1))

    subject_filter = _prepare_subject_filter(params, subject_files)
    print('Subject filter:', subject_filter)

    for subject_file in subject_files:
        subject_id = subject_file.split('_')[1].split('.')[0]
        if subject_id not in subject_filter:
            continue

        print('--------- Processing file:', subject_file, '---------')

        subject_params = _prepare_subject_params(params, subject_id)

        if getattr(params, "separate_models", False):
            results, conf_mat, labels = train_separate_targets(subject_params, seeds)
        else:
            results, conf_mat, labels = train_joint_targets(subject_params, seeds)

        save_and_plot_results(
            subject_params, results, conf_mat, labels,
            experiment_log_dir=params.log_dir
        )

    return params.log_dir

def _prepare_subject_params(base_params: Namespace, subject_id: str) -> Namespace:
    """Prepare parameters for a specific subject."""
    subject_params = Namespace(**vars(base_params))
    subject_params.subject_id = subject_id
    subject_params.sample_path = os.path.join(
        base_params.sample_dir, f'subject_{subject_id}.npz'
    )
    subject_params.log_dir = os.path.join(base_params.log_dir, f"subject_{subject_id}")
    subject_params.channel_file = os.path.join(
        base_params.channel_selection_dir, f'subject_{subject_id}.json'
    )
    os.makedirs(subject_params.log_dir, exist_ok=True)
    return subject_params


def _prepare_subject_filter(params: Namespace, subject_files: list) -> list:
    """Prepare a filter for subject files based on provided subject IDs."""
    if params.subject_ids:
        subject_filter = []
        for subject_id in params.subject_ids:
            if isinstance(subject_id, int):
                subject_filter.append(str(subject_id))
            else:
                subject_filter.append(subject_id)
    else:
        subject_filter = [
            f.replace('.npz', '').replace('subject_', '')
            for f in subject_files if f.startswith('subject_')
        ]

    return subject_filter


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train_classifier.py <config.yaml>")
    config_path = sys.argv[1]
    config = load_config(config_path)
    run(config)

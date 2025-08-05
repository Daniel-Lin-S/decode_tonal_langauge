"""Train a classifier on ECoG data using a YAML configuration."""

import os
import sys
import numpy as np
import torch

from training.classifier_pipeline import (
    train_joint_targets,
    train_separate_targets,
    save_and_plot_results,
)
from utils.config import load_config, dict_to_namespace


def run(config: dict, config_path: str | None = None) -> None:
    """Run classifier training from a configuration dictionary."""
    training_section = config.get("training", {})
    train_cfg = training_section.get("params", {})
    flat_train = {}
    for section in ("io", "experiment", "training"):
        flat_train.update(train_cfg.get(section, {}))
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})

    for key in ("model", "model_name", "foundation_weights_path"):
        if key in model_cfg:
            flat_train[key] = model_cfg[key]

    params = dict_to_namespace(flat_train)

    # Set default model name if not provided
    if getattr(params, "model_name", None) is None and "model" in model_cfg:
        params.model_name = model_cfg["model"].split(".")[-1]

    np.random.seed(getattr(params, "seed", 42))
    seeds = np.random.randint(0, 10000, getattr(params, "repeat", 1))

    eval_cfg = config.get("evaluation", {}).get("params", {})

    combined_cfg = {**dataset_cfg, **{k: v for k, v in model_cfg.items() if k not in ("model", "model_name", "foundation_weights_path")}}

    if getattr(params, "separate_models", False):
        results, conf_mat, labels = train_separate_targets(params, combined_cfg, seeds, eval_cfg)
    else:
        results, conf_mat, labels = train_joint_targets(params, combined_cfg, seeds, eval_cfg)

    save_and_plot_results(params, results, conf_mat, labels, eval_cfg)

    # Copy config to log directory for record keeping
    log_dir = getattr(params, "log_dir", None)
    if config_path is not None and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        import shutil
        shutil.copy(config_path, os.path.join(log_dir, "config_used.yaml"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train_classifier.py <config.yaml>")
    config_path = sys.argv[1]
    config = load_config(config_path)
    run(config, config_path)

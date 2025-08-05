"""High level training routines for classifier experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from data_loading.dataloaders import split_dataset
from data_loading.classifier_utils import prepare_erps_labels
from models.classifier_factory import get_classifier_by_name
from models.classifier_trainer import LightningClassifier
from utils.utils import set_seeds, prepare_class_labels
from utils.visualise import plot_confusion_matrix
from utils.metrics import compute_classification_metrics_joint


def train_separate_targets(
    params,
    seeds: np.ndarray,
    eval_cfg: Dict,
) -> Tuple[Dict, np.ndarray, List[str]]:
    """Train a separate classifier for each target and combine results."""

    class_labels = prepare_class_labels(
        params.targets,
        class_label_dict={
            "syllable": params.syllable_labels, "tone": params.tone_labels
        },
    )

    all_datasets: Dict[str, TensorDataset] = {}
    input_shapes: Dict[str, Tuple[int, int]] = {}
    channels = set()
    n_classes_dict: Dict[str, int] = {}

    for target in params.targets:
        erps, labels, chs, cls = prepare_erps_labels(
            params.sample_path, [target], params.channel_file)
        n_classes_dict[target] = cls[target]
        channels.update(chs)

        erps_tensor = torch.tensor(erps, dtype=torch.float32).to(params.device)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).to(params.device)
        all_datasets[target] = TensorDataset(erps_tensor, labels_tensor)
        input_shapes[target] = erps_tensor.shape[1:]

        if params.verbose > 0:
            print(
                f"Prepared {erps_tensor.shape[0]} samples with shape {erps_tensor.shape} "
                f"for target {target}"
            )

    n_classes = 1
    for cls in n_classes_dict.values():
        n_classes *= cls

    metrics = eval_cfg.get("metrics", ["accuracy"])
    metric_values: Dict[str, List[float]] = {m: [] for m in metrics if m != "confusion_matrix"}
    confusion_mat = np.zeros((n_classes, n_classes)) if "confusion_matrix" in metrics else None
    model_size = 0

    for i, seed in enumerate(seeds):
        set_seeds(int(seed))
        all_preds: Dict[str, np.ndarray] = {}
        all_true: Dict[str, np.ndarray] = {}

        for target, dataset in all_datasets.items():
            if params.verbose > 0:
                print(f"Training for target: {target} with seed {seed}...")

            loaders = split_dataset(
                dataset,
                [params.train_ratio, params.vali_ratio, params.test_ratio],
                shuffling=[True, False, False],
                batch_size=params.batch_size,
                seed=int(seed),
            )

            all_true[target] = np.concatenate([b[1].cpu().numpy() for b in loaders[2]])

            trainer_verbose = params.verbose > 1

            model = get_classifier_by_name(
                params.model,
                device=params.device,
                n_classes=n_classes_dict[target],
                n_channels=input_shapes[target][0],
                seq_length=input_shapes[target][1],
                classifier_kwargs=params.model_kwargs
            )
            model_size += model.get_nparams()

            model_verbose = params.verbose > 0 and i == 0
            if model_verbose:
                print(f"Number of trainable parameters: {model.get_layer_nparams()}")

            lightning_model = LightningClassifier(model, learning_rate=params.lr)

            accelerator = "gpu" if "cuda" in params.device else "cpu"
            devices = [int(params.device.split(":")[1])] if "cuda" in params.device else 1
            early_stop = EarlyStopping(
                monitor="val/loss", patience=params.patience, mode="min"
            )

            tb_logger = TensorBoardLogger(
                save_dir=os.path.join(params.log_dir, f"{params.model_name}_{target}_tb"),
                name=f"sub_{params.subject_id}",
                version='seed_'+str(seed)
            )
            csv_logger = CSVLogger(
                save_dir= os.path.join(params.log_dir, f"{params.model_name}_{target}_csv"),
                name=f"sub_{params.subject_id}",
                version='seed_'+str(seed)
            )

            trainer = pl.Trainer(
                max_epochs=params.epochs,
                logger=[tb_logger, csv_logger],
                enable_checkpointing=False,
                callbacks=[early_stop],
                accelerator=accelerator,
                devices=devices,
                enable_progress_bar=trainer_verbose,
                log_every_n_steps=params.log_every_n_steps
            )

            trainer.fit(lightning_model, loaders[0], loaders[1])

            trainer.test(lightning_model, loaders[2])
            preds = torch.cat(trainer.predict(lightning_model, loaders[2])).cpu().numpy()

            if params.model_dir is not None:
                save_path = os.path.join(
                    params.model_dir, f"{target}_{params.model_name}_seed_{seed}.pt"
                )
                torch.save(model.state_dict(), save_path)
                if params.verbose > 0:
                    print(f"Model saved to {save_path}")

            preds = torch.cat(trainer.predict(lightning_model, loaders[2])).cpu().numpy()
            all_preds[target] = preds.cpu().numpy()

        joint_metrics = compute_classification_metrics_joint(
            all_true, all_preds, metrics=metrics
        )

        for m in metrics:
            if m == "confusion_matrix":
                continue
            metric_values[m].append(joint_metrics[m])

        if confusion_mat is not None and "confusion_matrix" in joint_metrics:
            confusion_mat += joint_metrics["confusion_matrix"]

    result_info = {
        **metric_values,
        "model_size": model_size,
        "channels": sorted(channels),
        "class_labels": class_labels,
    }
    return result_info, confusion_mat, class_labels


def train_joint_targets(
    params,
    config: Dict,
    seeds: np.ndarray,
    eval_cfg: Dict,
) -> Tuple[Dict, np.ndarray, List[str]]:
    """Train a single model that predicts multiple targets jointly."""

    erps, labels, channels, n_classes_dict = prepare_erps_labels(
        params.sample_path, params.targets, params.channel_file
    )

    erps_tensor = torch.tensor(erps, dtype=torch.float32).to(params.device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(params.device)
    dataset = TensorDataset(erps_tensor, labels_tensor)
    n_samples, n_channels, seq_length = erps_tensor.shape

    if params.verbose > 0:
        print(
            f"Prepared {n_samples} samples with shape {erps_tensor.shape} "
            f"and labels with shape {labels_tensor.shape}"
        )

    n_classes = len(np.unique(labels))
    class_labels = prepare_class_labels(
        params.targets,
        n_classes_dict=n_classes_dict,
        class_label_dict={
            "syllable": params.syllable_labels, "tone": params.tone_labels
        },
    )

    metrics = eval_cfg.get("metrics", ["accuracy"])
    metric_values: Dict[str, List[float]] = {m: [] for m in metrics if m != "confusion_matrix"}
    confusion_mat = np.zeros((n_classes, n_classes)) if "confusion_matrix" in metrics else None
    model_size = 0

    for i, seed in enumerate(seeds):
        set_seeds(int(seed))

        loaders = split_dataset(
            dataset,
            [params.train_ratio, params.vali_ratio, params.test_ratio],
            shuffling=[True, False, False],
            batch_size=params.batch_size,
            seed=int(seed),
        )

        trainer_verbose = params.verbose > 1

        model = get_classifier_by_name(
            params.model, params.device,
            n_classes, n_channels, seq_length,
            classifier_kwargs=params.model_kwargs
        )
        model_size = model.get_nparams()

        model_verbose = params.verbose > 0 and i == 0
        if model_verbose:
            print(f"Number of trainable parameters: {model.get_layer_nparams()}")

        lightning_model = LightningClassifier(model, learning_rate=params.lr)

        accelerator = "gpu" if "cuda" in params.device else "cpu"
        devices = [int(params.device.split(":")[1])] if "cuda" in params.device else 1
        early_stop = EarlyStopping(
            monitor="val/loss", patience=params.patience, mode="min"
        )

        target_name = "_".join(params.targets) if len(params.targets) > 1 else params.targets[0]
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(params.log_dir, f"{params.model_name}_{target_name}_tb"),
            name=f"sub_{params.subject_id}",
            version='seed_'+str(seed)
        )

        csv_logger = CSVLogger(
            save_dir=os.path.join(params.log_dir, f"{params.model_name}_{target_name}_csv"),
            name=f"sub_{params.subject_id}",
            version='seed_'+str(seed)
        )

        trainer = pl.Trainer(
            max_epochs=params.epochs,
            logger=[tb_logger, csv_logger],
            enable_checkpointing=False,
            callbacks=[early_stop],
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=trainer_verbose,
            log_every_n_steps=params.log_every_n_steps
        )

        trainer.fit(lightning_model, loaders[0], loaders[1])

        trainer.test(lightning_model, loaders[2])

        preds = torch.cat(trainer.predict(lightning_model, loaders[2])).cpu().numpy()
        true = np.concatenate([b[1].cpu().numpy() for b in loaders[2]])

        joint_metrics = compute_classification_metrics_joint(
            {"label": true}, {"label": preds}, metrics=metrics
        )

        if params.model_dir is not None:
            target_str = "_".join(params.targets) if len(params.targets) > 1 else params.targets[0]
            save_path = os.path.join(
                params.model_dir, f"{target_str}_{params.model_name}_seed_{seed}.pt"
            )
            torch.save(model.state_dict(), save_path)
            if params.verbose > 0:
                print(f"Model saved to {save_path}")

        if confusion_mat is not None and "confusion_matrix" in joint_metrics:
            confusion_mat += joint_metrics["confusion_matrix"]

        for m in metrics:
            if m == "confusion_matrix":
                continue
            metric_values[m].append(joint_metrics[m])

    result_info = {
        **metric_values,
        "model_size": model_size,
        "channels": channels,
        "class_labels": class_labels,
        "seeds": seeds.tolist(),
    }
    return result_info, confusion_mat, class_labels


def save_and_plot_results(
    params,
    result_info: Dict,
    confusion_matrix: np.ndarray,
    class_labels: List[str],
    eval_cfg: Dict,
) -> None:
    """Save results to CSV and generate plots."""

    metrics = eval_cfg.get("metrics", [])
    aggregates = eval_cfg.get("aggregates", ["mean", "std"])
    if isinstance(aggregates, str):
        aggregates = [aggregates]

    results = {
        "model_name": params.model_name,
        "model_kwargs": str(params.model_kwargs) if hasattr(params, "model_kwargs") else "{}",
        "model_size": result_info["model_size"],
        "subject": params.subject_id,
        "target": ",".join(params.targets),
        "electrodes": str(result_info["channels"]),
        "seeds": str(result_info.get("seeds"))
    }

    for m in metrics:
        if m == "confusion_matrix":  # already aggregated
            continue
        values = result_info[m]
        for agg in aggregates:
            try:
                agg_func = getattr(np, agg)
            except AttributeError:
                raise ValueError(
                    f"Aggregate function '{agg}' is not recognized in numpy."
                    "Please change evaluation.aggregates parameter."
                )
            results[f"{m}_{agg}"] = float(agg_func(values))
        results[f"{m}_all"] = str(values)

    if confusion_matrix is not None and "confusion_matrix" in metrics:
        results["confusion_matrix"] = str(confusion_matrix.tolist())

    df = pd.DataFrame([results])
    if os.path.exists(params.result_file):
        df.to_csv(params.result_file, mode="a", header=False, index=False)
    else:
        df.to_csv(params.result_file, index=False)
    print(f"Results saved to {params.result_file}")

    if confusion_matrix is not None and "confusion_matrix" in metrics:
        add_numbers = confusion_matrix.shape[0] <= 10
        plot_confusion_matrix(
            confusion_matrix,
            add_numbers,
            label_names=class_labels,
            figure_path=os.path.join(params.figure_dir, "confusion_matrix.png"),
        )
        print(f"Confusion matrix saved to {params.figure_dir}/confusion_matrix.png")

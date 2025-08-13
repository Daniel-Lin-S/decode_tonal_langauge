"""High level training routines for classifier experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple
from argparse import Namespace

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from data_loading.dataloaders import split_dataset
from data_loading.sample_loading import ClassificationSampleHandler
from models.classifier_factory import get_classifier_by_name
from models.classifier_trainer import LightningClassifier
from utils.utils import set_seeds
from utils.visualise import plot_confusion_matrix
from utils.metrics import (
    compute_classification_metrics_joint, compute_classification_metrics
)


def train_separate_targets(
    params: Namespace,
    seeds: np.ndarray
) -> Tuple[Dict, np.ndarray, List[str]]:
    """Train a separate classifier for each target and combine results."""
    verbose = getattr(params, "verbose", 1)

    all_datasets: Dict[str, TensorDataset] = {}
    input_shapes: Dict[str, Tuple[int, int]] = {}
    channels: Dict[str, List[int]] = {}
    n_classes_dict: Dict[str, int] = {}

    for target in params.targets:
        target_params = Namespace(**vars(params))
        target_params.targets = [target]
        target_handler = ClassificationSampleHandler(target_params)
        data = target_handler.load_data()

        features = data['features']
        n_classes_dict.update({target: data['n_classes_dict'][target]})
        channels[target] = data['selected_channels']

        all_datasets[target] = target_handler.prepare_torch_dataset(
            features, data['labels'], params.device)

        input_shapes[target] = features.shape[1:]

        if verbose > 0:
            print(
                f"Prepared {features.shape[0]} samples with shape {features.shape} "
                f"for target {target}"
            )

    class_labels = ClassificationSampleHandler(params).prepare_class_labels(data['n_classes_dict'])

    n_classes = 1
    for cls in n_classes_dict.values():
        n_classes *= cls

    metrics = getattr(params, "metrics", ["accuracy"])
    metric_values: Dict[str, List[float]] = {m: [] for m in metrics if m != "confusion_matrix"}
    confusion_mat = np.zeros((n_classes, n_classes)) if "confusion_matrix" in metrics else None
    model_size = 0

    individual_metrics = {
        target: {m: [] for m in metrics if m != "confusion_matrix"}
        for target in params.targets
    }
    individual_confusion_mat = {
        target: np.zeros((n_classes_dict[target], n_classes_dict[target]))
        for target in params.targets
    } if "confusion_matrix" in metrics else None

    for i, seed in enumerate(seeds):
        set_seeds(int(seed))
        all_preds: Dict[str, np.ndarray] = {}
        all_true: Dict[str, np.ndarray] = {}

        for target, dataset in all_datasets.items():
            if verbose > 1:
                print(f"Training for target: {target} with seed {seed}...")

            loaders = split_dataset(
                dataset,
                [params.train_ratio, params.vali_ratio, params.test_ratio],
                shuffling=[True, False, False],
                batch_size=params.batch_size,
                seed=int(seed),
            )

            all_true[target] = np.concatenate([b[1].cpu().numpy() for b in loaders[2]])

            trainer_verbose = verbose > 1

            model = get_classifier_by_name(
                params.model,
                device=params.device,
                n_classes=n_classes_dict[target],
                n_channels=input_shapes[target][0],
                seq_length=input_shapes[target][1],
                classifier_kwargs=params.model_kwargs
            )
            model_size += model.get_nparams()

            model_verbose = verbose > 0 and i == 0
            if model_verbose:
                print(f"Number of trainable parameters: {model.get_layer_nparams()}")

            lightning_model = LightningClassifier(model, learning_rate=params.lr)
            lightning_model.verbose = model_verbose

            accelerator = "gpu" if "cuda" in params.device else "cpu"
            devices = [int(params.device.split(":")[1])] if "cuda" in params.device else 1
            early_stop = EarlyStopping(
                monitor="val/loss", patience=params.patience, mode="min"
            )

            tb_logger = TensorBoardLogger(
                save_dir=os.path.join(params.log_dir, f"{params.model_name}_{target}_tb"),
                name=f"subject_{params.subject_id}",
                version='seed_'+str(seed)
            )
            csv_logger = CSVLogger(
                save_dir= os.path.join(params.log_dir, f"{params.model_name}_{target}_csv"),
                name=f"subject_{params.subject_id}",
                version='seed_'+str(seed)
            )

            trainer = pl.Trainer(
                max_epochs=params.epochs,
                logger=[tb_logger, csv_logger],
                enable_checkpointing=False,
                enable_model_summary=model_verbose,
                callbacks=[early_stop],
                accelerator=accelerator,
                devices=devices,
                enable_progress_bar=trainer_verbose,
                log_every_n_steps=params.log_every_n_steps
            )

            trainer.fit(lightning_model, loaders[0], loaders[1])
            trainer.test(lightning_model, loaders[2])

            if params.save_checkpoints:
                model_dir = os.path.join(params.log_dir, "model_checkpoints")
                save_path = os.path.join(
                    model_dir, f"{target}_{params.model_name}_seed_{seed}.pt"
                )
                torch.save(model.state_dict(), save_path)
                if verbose > 0:
                    print(f"Model saved to {save_path}")

            preds = torch.cat(trainer.predict(lightning_model, loaders[2])).cpu().numpy()
            all_preds[target] = preds

            target_metrics = compute_classification_metrics(
                all_true[target], all_preds[target], metrics=metrics
            )

            for m in metrics:
                if m == "confusion_matrix":
                    continue
                individual_metrics[target][m].append(target_metrics[m])
            
            if individual_confusion_mat is not None and "confusion_matrix" in target_metrics:
                individual_confusion_mat[target] += target_metrics["confusion_matrix"]

        joint_metrics = compute_classification_metrics_joint(
            all_true, all_preds, metrics=metrics,
            verbose=verbose > 1
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
        "channels": channels,
        "seeds": seeds.tolist(),
        "class_labels": class_labels,
        "individual_metrics": individual_metrics,
        "individual_confusion_matrix": individual_confusion_mat
    }

    return result_info, confusion_mat, class_labels


def train_joint_targets(
    params: Namespace,
    seeds: np.ndarray
) -> Tuple[Dict, np.ndarray, List[str]]:
    """Train a single model that predicts multiple targets jointly."""
    verbose = getattr(params, "verbose", 1)

    data_handler = ClassificationSampleHandler(params)

    data = data_handler.load_data()
    dataset = data_handler.prepare_torch_dataset(
        data['features'], data['labels'],
        params.device
    )
    n_samples, n_channels, seq_length = data['features'].shape

    if verbose > 0:
        print(
            f"Prepared {n_samples} samples with shape {data['features'].shape} "
            f"and labels with shape {data['labels'].shape}"
        )

    n_classes = len(np.unique(data['labels']))
    class_labels = data_handler.prepare_class_labels(data['n_classes_dict'])

    metrics = getattr(params, "metrics", ["accuracy"])
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

        trainer_verbose = verbose > 1

        model = get_classifier_by_name(
            params.model, params.device,
            n_classes, n_channels, seq_length,
            classifier_kwargs=params.model_kwargs
        )
        model_size = model.get_nparams()

        model_verbose = verbose > 0 and i == 0
        if model_verbose:
            print(f"Number of trainable parameters: {model.get_layer_nparams()}")

        lightning_model = LightningClassifier(model, learning_rate=params.lr)
        lightning_model.verbose = model_verbose

        accelerator = "gpu" if "cuda" in params.device else "cpu"
        devices = [int(params.device.split(":")[1])] if "cuda" in params.device else 1
        early_stop = EarlyStopping(
            monitor="val/loss", patience=params.patience, mode="min"
        )

        target_name = "_".join(params.targets) if len(params.targets) > 1 else params.targets[0]
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(params.log_dir, f"{params.model_name}_{target_name}_tb"),
            name=f"subject_{params.subject_id}",
            version='seed_'+str(seed)
        )

        csv_logger = CSVLogger(
            save_dir=os.path.join(params.log_dir, f"{params.model_name}_{target_name}_csv"),
            name=f"subject_{params.subject_id}",
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

        joint_metrics = compute_classification_metrics(
            true, preds, metrics=metrics,
            verbose=verbose > 1
        )

        if params.save_checkpoints:
            model_dir = os.path.join(params.log_dir, "model_checkpoints")
            target_str = ("_".join(params.targets)
                          if len(params.targets) > 1 else params.targets[0])
            save_path = os.path.join(
                model_dir, f"{target_str}_{params.model_name}_seed_{seed}.pt"
            )
            torch.save(model.state_dict(), save_path)
            if verbose > 0:
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
        "channels": data['selected_channels'],
        "class_labels": class_labels,
        "seeds": seeds.tolist(),
    }
    return result_info, confusion_mat, class_labels


def save_and_plot_results(
    params: Namespace,
    result_info: Dict,
    confusion_matrix: np.ndarray,
    class_labels: List[str]
) -> None:
    """Save results to CSV and generate plots."""

    metrics = getattr(params, "metrics", ["accuracy"])
    aggregates = getattr(params, "aggregates", ["mean", "std"])

    if isinstance(aggregates, str):
        aggregates = [aggregates]

    joint_label = ", ".join(getattr(params, "targets", []))

    # collect channels
    def _norm_channel_list(chs) -> list[int]:
        if chs is None:
            return []
        # flatten-like but expect already flat lists
        return sorted({int(c) for c in chs})

    def _channels_for(target_label: str) -> str:
        """
        Return a CSV string of channels for a given target label.
        - If result_info['channels'] is list[int]: same for all targets (and joint).
        - If dict[str, list[int]]: pick per-target; for joint -> union over params.targets.
        """
        chs_info = result_info.get("channels", [])
        # List[int] → directly use for all rows
        if isinstance(chs_info, (list, tuple, np.ndarray)):
            chs = _norm_channel_list(chs_info)
            return ",".join(map(str, chs))
        # Dict[str, List[int]]
        if isinstance(chs_info, dict):
            if target_label == joint_label:
                # union over declared targets
                union = set()
                for t in getattr(params, "targets", []):
                    lst = chs_info.get(str(t), [])
                    union.update(int(c) for c in lst)
                chs = sorted(union)
            else:   # individual target
                chs = _norm_channel_list(chs_info.get(str(target_label), []))
            return ",".join(map(str, chs))

        return ""

    def _build_row(
            metric_dict: Dict[str, list], target_label: str
        ) -> Dict[str, object]:
        """Create one CSV row from a dict: metric -> list[float] across seeds."""
        row = {
            "model_name": params.model_name,
            "model_size": result_info.get("model_size"),
            "subject": params.subject_id,
            "target": target_label,
            "channels": _channels_for(target_label),
            "seeds": str(result_info.get("seeds")),
        }
        for m in metrics:
            if m == "confusion_matrix":
                continue
            values = metric_dict.get(m, [])
            for agg in aggregates:
                agg_func = getattr(np, agg, None)
                if agg_func is None:
                    raise ValueError(
                        f"Aggregate function '{agg}' is not recognized in numpy. "
                        "Please change evaluation.aggregates parameter."
                    )
                row[f"{m}_{agg}"] = float(agg_func(values)) if len(values) else np.nan
            row[f"{m}_all"] = str(list(values))

        return row

    rows = []

    joint_metric_dict = {m : result_info[m] for m in metrics if m != "confusion_matrix"}
    rows.append(_build_row(joint_metric_dict, ", ".join(params.targets)))

    if "individual_metrics" in result_info:
        individual_metrics = result_info["individual_metrics"]
        for target, metrics_dict in individual_metrics.items():
            rows.append(_build_row(metrics_dict, str(target)))

    df = pd.DataFrame(rows)

    result_path = os.path.join(params.log_dir, "results.csv")
    if os.path.exists(result_path):
        df.to_csv(result_path, mode="a", header=False, index=False)
    else:
        df.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")

    figure_dir = os.path.join(params.log_dir, f"figures/subject_{params.subject_id}")
    os.makedirs(figure_dir, exist_ok=True)

    cm_dir = os.path.join(params.log_dir, f"confusion_matrices/subject_{params.subject_id}")
    os.makedirs(cm_dir, exist_ok=True)

    if confusion_matrix is not None and "confusion_matrix" in metrics:
        add_numbers = confusion_matrix.shape[0] <= 10
        plot_confusion_matrix(
            confusion_matrix,
            add_numbers,
            label_names=class_labels,
            figure_path=os.path.join(figure_dir, "confusion_matrix.png"),
        )
        print(f"Confusion matrix saved to {figure_dir}/confusion_matrix.png")

        cm_csv_path = os.path.join(cm_dir, "confusion_matrix.csv")
        pd.DataFrame(confusion_matrix).to_csv(cm_csv_path, index=False)

    if "individual_confusion_matrix" in result_info:
        individual_confusion_matrices = result_info["individual_confusion_matrix"]
        for target, cm in individual_confusion_matrices.items():

            cm_figure_path = os.path.join(
                cm_dir, f"confusion_matrix_{target}.png"
            )

            if cm is not None:
                add_numbers = cm.shape[0] <= 10
                plot_confusion_matrix(
                    cm,
                    add_numbers,
                    label_names=class_labels,
                    figure_path=cm_figure_path,
                )
                print(f"Confusion matrix for {target} saved to {cm_figure_path}")

            cm_csv_path = os.path.join(figure_dir, f"confusion_matrix_{target}.csv")
            pd.DataFrame(cm).to_csv(cm_csv_path, index=False)

from typing import List, Dict, Optional
from argparse import Namespace
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset


class ClassificationSampleHandler:
    """
    Handles data loading, preprocessing, and dataset preparation
    from samples stored in `.npz` files.
    Each sample should be 2-dimensional, with the first dimension
    representing channels (to be filtered based on a channel file).
    """
    sample_path: str
    channel_file: str | None
    targets: List[str]

    def __init__(
            self, params: Namespace
        ):
        """Initialize the DataHandler."""
        self.sample_path = params.sample_path
        self.channel_file = params.channel_file if hasattr(params, 'channel_file') else None
        self.dataset = np.load(self.sample_path)
        self.channels = None
        self.targets = getattr(params, 'targets', None)
        if isinstance(self.targets, str):
            self.targets = [self.targets]

        self.params = params

    def load_data(self) -> dict:
        """
        Load data, labels, and channel selections.

        Returns
        -------
        dict:
            - features (np.ndarray): Input data (n_samples, n_channels, n_timepoints).
            - labels (np.ndarray): Labels corresponding to the data (n_samples,).
            - select_channels (np.ndarray): Indices of the selected channels.
            - n_classes_dict (dict): Dictionary mapping targets to the number of classes.
        """

        try:
            features = self.dataset[self.params.features]
        except KeyError:
            raise KeyError(
                f"The dataset in {self.sample_path} does not contain {self.params.input_key}. "
                f"Available keys: {', '.join(self.dataset.keys())}"
            )

        target_labels = []
        n_classes_dict = {}
        for target in self.targets:
            if target not in self.dataset:
                raise KeyError(
                    f"The dataset does not contain '{target}' key. "
                    f"Available keys: {', '.join(self.dataset.keys())}"
                )

            target_labels.append(self.dataset[target].flatten())
            n_classes_dict[target] = len(np.unique(self.dataset[target]))

        # Combine target labels into a single label array
        labels = np.zeros_like(target_labels[0], dtype=int)
        multiplier = 1
        for target_label in target_labels:
            labels += target_label * multiplier
            multiplier *= len(np.unique(target_label))

        # Filter channels if a channel file is provided
        self.channels = self._filter_channels(features.shape[1])
        features = features[:, self.channels, :]

        data = {
            'features': features,
            'labels': labels,
            'selected_channels': self.channels,
            'n_classes_dict': n_classes_dict
        }

        return data

    def _filter_channels(self, n_channels: int) -> np.ndarray:
        """
        Filter channels based on the channel file.

        Args:
            n_channels (int): Total number of channels in the dataset.

        Returns:
            np.ndarray: Indices of the selected channels.
        """
        if self.channel_file is None:
            return np.arange(n_channels)

        with open(self.channel_file, "r") as f:
            channel_selections = json.load(f)

        channels = set()
        for target in self.targets:
            key = f"{target}_discriminative"
            if key not in channel_selections:
                raise KeyError(
                    f"Channel selection for '{key}' not found in the file {self.channel_file}. "
                    f"Available keys: {', '.join(channel_selections.keys())}"
                )
            channels.update(channel_selections[key])

        if not channels:
            raise ValueError(
                f"No channels found for the targets: {', '.join(self.targets)}. "
                f"Please check the channel file {self.channel_file}"
            )

        return np.array(sorted(channels))


    def prepare_torch_dataset(
        self, features: np.ndarray, labels: np.ndarray, device: str
    ) -> TensorDataset:
        """
        Prepare a PyTorch dataset from ECoG data and labels.

        Args:
            features (np.ndarray): Input data (n_samples, n_channels, n_timepoints).
            labels (np.ndarray): Labels corresponding to the data (n_samples,).
            device (str): Device to move the data to (e.g., "cuda" or "cpu").

        Returns:
            TensorDataset: A PyTorch dataset containing the ECoG data and labels.
        """
        ecog_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
        return TensorDataset(ecog_tensor, labels_tensor)

    def prepare_class_labels(
            self,
            n_classes_dict: Optional[Dict[str, int]]=None
        ) -> List[str]:
        """
        Prepare class labels for the targets based on the number of classes
        and optional class label mappings.

        Parameters
        ----------
        n_classes_dict : Dict[str, int]
            Dictionary mapping target names to the number of classes.
        """
        class_labels_dict = getattr(self.params, 'class_labels', {})
        print('Class label mapping: ', class_labels_dict)

        if len(self.targets) > 1:
            class_labels = []
            for target in self.targets:
                if target not in class_labels_dict or class_labels_dict[target] is None:
                    if target not in n_classes_dict:
                        raise ValueError(
                            f"Number of classes for target '{target}' is not provided."
                        )
                    # set default labels
                    class_labels.append(
                        np.arange(1, n_classes_dict[target] + 1).astype(str)
                    )
                else:
                    class_labels.append(class_labels_dict[target])

            # Generate Cartesian product of class labels for all targets
            from itertools import product
            class_labels = [
                '_'.join(label_combination)
                for label_combination in product(*class_labels)
            ]

        else:  # Handle a single target
            target = self.targets[0]
            if n_classes_dict is None:
                raise ValueError(
                    f"Number of classes for target '{target}' is not provided."
                )
            if class_labels_dict[target] is None:
                if target not in n_classes_dict:
                    raise ValueError(
                        f"Number of classes for target '{target}' is not provided."
                    )
                class_labels = np.arange(
                    1, n_classes_dict[target] + 1).astype(str)
            else:
                class_labels = class_labels_dict[target]

        return class_labels

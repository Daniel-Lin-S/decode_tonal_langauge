"""Build classifier models for the training scripts using dynamic imports."""

from typing import Dict
from importlib import import_module
import inspect

from .classifier import ClassifierModel


def get_classifier_by_name(
        model_path: str,
        device: str,
        n_classes: int,
        n_channels: int,
        seq_length: int,
        classifier_kwargs: Dict={}
    ) -> ClassifierModel:
    """Dynamically import and build a classifier model.

    Parameters
    ----------
    model_path : str
        Full python path to the classifier class.
    device : str
        Device to place the model on.
    n_classes : int
        Number of output classes.
    n_channels : int
        Number of input channels.
    seq_length : int
        Length of the input sequence.
    classifier_kwargs : Dict | None
        Additional keyword arguments passed to the model constructor.
    """
    classifier_kwargs = classifier_kwargs or {}

    module_name, class_name = model_path.rsplit('.', 1)
    module = import_module(module_name)
    cls = getattr(module, class_name)

    base_kwargs = {
        'n_classes': n_classes,
        'n_channels': n_channels,
        'seq_length': seq_length,
        'input_channels': n_channels,
        'input_length': seq_length,
        'input_dim': n_channels * seq_length,
    }

    if classifier_kwargs:
        base_kwargs.update(classifier_kwargs)

    # Filter kwargs based on the model signature
    sig = inspect.signature(cls)
    allowed_kwargs = {k: v for k, v in base_kwargs.items() if k in sig.parameters}

    model = cls(**allowed_kwargs).to(device)
    return model

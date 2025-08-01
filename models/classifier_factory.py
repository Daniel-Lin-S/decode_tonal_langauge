"""Build classifier models for the training scripts."""

from typing import Dict

from .classifier import ClassifierModel
from .simple_classifiers import (
    LogisticRegressionClassifier,
    ShallowNNClassifier
)
from .deep_classifiers import (
    CNNClassifier,
    CNNRNNClassifier
)
from .cbramod_classifier import CBraModClassifier


MODEL_CHOICES = ['logistic', 'CNN', 'CNN-RNN', 'ShallowNN', 'CBraMod']


def get_classifier_by_name(
        model_name: str,
        device: str,
        n_classes: int,
        n_channels: int,
        seq_length: int,
        classifier_kwargs: Dict | None = None
    ) -> ClassifierModel:
    """
    Build the classifier model based on the specified parameters.

    Parameters
    ----------
    model_name : str
        The name of the model to build. 
        Choices are 'logistic', 'CNN', 'CNN-RNN', 'ShallowNN', 'CBraMod'.
    device : str
        The device to use for training ('cpu' or 'cuda').
        If 'cuda', it will use the first available GPU.
        If 'cpu', it will use the CPU.
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

    if model_name == 'logistic':
        model = LogisticRegressionClassifier(
                    input_dim=n_channels * seq_length,
                    n_classes=n_classes,
                    **classifier_kwargs
                ).to(device)
    elif model_name == 'CNN':
        model = CNNClassifier(
                    input_channels=n_channels,
                    input_length=seq_length,
                    n_classes=n_classes,
                    **classifier_kwargs
                ).to(device)
    elif model_name == 'ShallowNN':
        model = ShallowNNClassifier(
                    input_dim=n_channels * seq_length,
                    n_classes=n_classes,
                    **classifier_kwargs
                ).to(device)
    elif model_name == 'CNN-RNN':
        model = CNNRNNClassifier(
                    input_channels=n_channels,
                    input_length=seq_length,
                    n_classes=n_classes,
                    **classifier_kwargs
                ).to(device)
    elif model_name == 'CBraMod':
        model = CBraModClassifier(
                    input_channels=n_channels,
                    input_length=seq_length,
                    n_classes=n_classes,
                    device=device,
                    **classifier_kwargs
                ).to(device)
    else:
        raise ValueError(
                    f"Invalid model name '{model_name}'. "
                    f"Choose from {MODEL_CHOICES}."
                )
        
    return model
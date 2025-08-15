"""
Please download pre-trained CBraMod weights from https://huggingface.co/weighting666/CBraMod
"""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from scipy.signal import resample

from .foundation.CBraMod import CBraMod
from .classifier import ClassifierModel


class CBraModClassifier(ClassifierModel):
    """
    A classifier using the CBraMod backbone
    and a 2-layer fully connected layer for classification.

    This model must be trained with cross-entropy loss.
    """
    def __init__(
            self, n_classes: int,
            input_channels: int,
            input_length: int,
            input_sampling_rate: int = 200,
            use_pretrained_weights: bool = True,
            pretrained_weights_path: str = None,
            backbone_kwargs: dict = {},
            activation: str='ELU',
            device: str = 'cpu'
        ):
        """
        Parameters
        ----------
        n_classes : int
            Number of classes for classification.
        input_channels : int
            Number of input channels (e.g., EEG channels).
        input_length : int
            Length of the input time series (number of timepoints).
        input_sampling_rate : int, optional
            Sampling rate of the input data, by default 200.
            If the sampling rate is not 200Hz,
            the input will be resampled to 200Hz to align
            with the CBraMod backbone requirements.
        use_pretrained_weights : bool, optional
            Whether to use pretrained weights, by default True.
        pretrained_weights_path : str, optional
            Path to the pretrained weights file, by default None.
        backbone_kwargs : dict, optional
            Additional keyword arguments for the CBraMod backbone, by default {}.
            Warning: if you use different settings from the default,
            the published weights will not work.
        activation : str, optional
            Activation function to use in the classifier, by default 'ELU'.
        device : str, optional
            Device to load the backbone model onto, by default 'cpu'.
        """
        super(CBraModClassifier, self).__init__(n_classes)
        self.backbone = CBraMod(**backbone_kwargs)
        self.input_sampling_rate = input_sampling_rate

        self.in_dim = backbone_kwargs.get('in_dim', 200)

        if use_pretrained_weights:
            if pretrained_weights_path is None:
                raise ValueError(
                    "Pretrained weights path must be provided "
                    "if use_pretrained_weights is True."
                )
            self.backbone.load_state_dict(
                torch.load(pretrained_weights_path, map_location=device),
            )

        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(input_channels * input_length, input_length),
            _get_activation(activation),
            nn.Dropout(),
            nn.Linear(input_length, n_classes)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, timepoints).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_classes).
            The logits for each class and each batch.
        """
        if self.input_sampling_rate != self.in_dim:
            # Resample the input to match the required sampling rate
            num_samples = int(x.shape[2] * self.in_dim / self.input_sampling_rate)
            x = resample(x, num_samples, axis=2)

        # cut into patches
        x = x.view(x.shape[0], x.shape[1], -1, self.in_dim)

        features = self.backbone.forward(x)

        return self.classifier(features)


def _get_activation(activation: str) -> nn.Module:
    """
    Get the activation function based on the provided name.

    Parameters
    ----------
    activation : str
        Name of the activation function.

    Returns
    -------
    nn.Module
        The corresponding activation function module.
    """
    if activation == 'ELU':
        return nn.ELU()
    elif activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU()
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'GLU':
        return nn.GLU()
    elif activation == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(
            f"Unsupported activation function: {activation}")

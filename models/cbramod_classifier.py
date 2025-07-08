"""
Please download pre-trained CBraMod weights from https://huggingface.co/weighting666/CBraMod
"""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .foundation.CBraMod import CBraMod
from .classifier import ClassifierModel


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
            use_pretrained_weights: bool = True,
            pretrained_weights_path: str = None,
            backbone_kwargs: dict = {},
            activation: str='ELU'
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
        """
        super(CBraModClassifier, self).__init__(n_classes)
        self.backbone = CBraMod(**backbone_kwargs)

        self.in_dim = backbone_kwargs.get('in_dim', 200)

        if use_pretrained_weights:
            if pretrained_weights_path is None:
                raise ValueError(
                    "Pretrained weights path must be provided "
                    "if use_pretrained_weights is True."
                )
            self.backbone.load_state_dict(torch.load(pretrained_weights_path))

        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Sequential(
            Rearrange('b c s d - >b (c s d)'),
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
        if x.shape[2] % self.in_dim != 0:
            raise ValueError(
                "Input tensor's time dimension must be a multiple of in_dim."
                f" Received input shape: {x.shape}, in_dim: {self.in_dim}"
            )

        # cut into segments
        x = x.view(x.shape[0], x.shape[1], -1, 200)

        features = self.backbone.forward(x)
        return self.classifier(features)

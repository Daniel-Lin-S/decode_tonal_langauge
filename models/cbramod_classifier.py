"""
Please download pre-trained CBraMod weights from https://huggingface.co/weighting666/CBraMod
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
import importlib
from inspect import signature

from .foundation.CBraMod import CBraMod
from .classifier import ClassifierModel
from .utils import get_activation


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
            freeze_backbone: bool = False,
            backbone_kwargs: dict = {},
            device: str = 'cpu',
            classification_head: str=None,
            classification_head_kwargs: dict = {}
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
        freeze_backbone : bool, optional
            Whether to freeze the backbone weights, by default False.
            If True, the backbone weights will not be updated during training.
        backbone_kwargs : dict, optional
            Additional keyword arguments for the CBraMod backbone, by default {}.
            Warning: if you use different settings from the default,
            the published weights will not work.
        activation : str, optional
            Activation function to use in the classifier, by default 'ELU'.
        device : str, optional
            Device to load the backbone model onto, by default 'cpu'.
        classification_head : str, optional
            The full path to a custom classification head class,
            if not given, a default linear layer will be used.
        classification_head_kwargs : dict, optional
            Additional keyword arguments for the classification head,
            by default {}.
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

        self._compute_effective_input_length(input_length)

        if classification_head:
            module_name, class_name = classification_head.rsplit('.', 1)
            module = importlib.import_module(module_name)
            ClassifierHead = getattr(module, class_name)

            constructor_args = signature(ClassifierHead.__init__).parameters

            self._initialise_classification_head_kwargs(
                input_channels, self.effective_input_length,
                classification_head_kwargs, constructor_args
            )

            self.classifier = ClassifierHead(
                n_classes=n_classes,
                **classification_head_kwargs
            )
        else:
            self.classifier = nn.Linear(
                self.effective_input_length * input_channels, n_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

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

        # pad when necessary
        timepoints = x.shape[2]
        remainder = timepoints % self.in_dim
        if remainder != 0:
            pad_size = self.in_dim - remainder
            x = F.pad(x, (0, pad_size))

        # cut into patches
        x = x.view(x.shape[0], x.shape[1], -1, self.in_dim)

        features = self.backbone.forward(x)

        features = features.view(features.shape[0], -1)

        return self.classifier(features)

    def _initialise_classification_head_kwargs(
            self, input_channels, input_length,
            classification_head_kwargs, constructor_args
        ):
        if 'hidden_dim' in constructor_args and (
                'hidden_dim' not in classification_head_kwargs
            ):
            classification_head_kwargs['hidden_dim'] = input_length
        if 'input_dim' in constructor_args:
            classification_head_kwargs['input_dim'] = input_length * input_channels
        if 'n_classes' in constructor_args:
            del classification_head_kwargs['n_classes']
        if 'input_channels' in constructor_args:
            classification_head_kwargs['input_channels'] = input_channels
        if 'input_length' in constructor_args:
            classification_head_kwargs['input_length'] = input_length

    def _compute_effective_input_length(self, input_length: int) -> int:
        """
        Compute the effective input length after padding.

        Parameters
        ----------
        input_length : int
            Original length of the input time series.

        Returns
        -------
        int
            Effective input length after padding.
        """
        remainder = input_length % self.in_dim
        if remainder == 0:
            self.effective_input_length = input_length
        else:
            self.effective_input_length = input_length + (self.in_dim - remainder)

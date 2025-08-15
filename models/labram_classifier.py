import torch
import torch.nn as nn
from scipy.signal import resample
from torch.nn.init import trunc_normal_
from typing import Optional, Sequence, Union
from inspect import signature

from .foundation import LaBraM
from .classifier import ClassifierModel
from .utils import load_state_dict


class LaBraMClassifier(ClassifierModel):
    def __init__(
            self, n_classes: int,
            input_length: int,
            input_channels: int,
            input_sampling_rate: int = 200,
            input_channel_names: Optional[Sequence[Union[str, int]]]=None,
            backbone_kwargs: dict = {},
            use_pretrained_weights: bool = True,
            pretrained_weights_path: Optional[str] = None,
            freeze_backbone: bool = False,
            init_scale: float=0.001,
            device: str = 'cpu',
            classification_head: str=None,
            classification_head_kwargs: dict = {}
        ):
        """
        Parameters
        ----------
        n_classes : int
            Number of classes for classification.
        input_length : int
            Length of the input time series (number of timepoints).
        input_channels : int
            Number of input channels (e.g., number of EEG electrodes).
        input_sampling_rate : int, optional
            Sampling rate of the input data, by default 200.
            If the sampling rate is not aligned with the patch size,
            (default 200Hz), the input will be resampled to match
            the LaBraM backbone.
        input_channel_names : Optional[Sequence[Union[str, int]]], optional
            Names of the input channels, used for the channel embedding
            of LaBraM.
        backbone_kwargs : dict, optional
            Additional keyword arguments for the LaBraM backbone, by default {}.
            Warning: if you use different settings from the default,
            the published weights may not work, some parameters have to match
            the original LaBraM model, such as patch_size.
        use_pretrained_weights : bool, optional
            Whether to use pretrained weights for the backbone, by default True.
        pretrained_weights_path : str, optional
            Path to the pretrained weights file, by default None.
            If a URL starting with "https" is provided, it will be downloaded.
        freeze_backbone : bool, optional
            Whether to freeze the backbone weights, by default False.
            If True, the backbone weights will not be updated during training.
        init_scale : float, optional
            Scale for the initial weights of the classifier head, by default 0.001.
        device: str, optional
            Device to load the pretrained weights onto, by default 'cpu'.
        classification_head : str, optional
            The full path to a custom classification head class,
            if not given, a default linear layer will be used.
        classification_head_kwargs : dict, optional
            Additional keyword arguments for the classification head,
            by default {}.
        """
        super(LaBraMClassifier, self).__init__(n_classes)
        self.backbone = LaBraM(EEG_size=input_length, **backbone_kwargs)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if use_pretrained_weights:
            self._load_backbone(pretrained_weights_path, device)

        self.input_sampling_rate = input_sampling_rate
        self.patch_size = backbone_kwargs.get('patch_size', 200)

        if classification_head:
            module_path, class_name = classification_head.rsplit('.', 1)
            head_module = __import__(module_path, fromlist=[class_name])
            head_class = getattr(head_module, class_name)

            constructor_args = signature(head_class.__init__).parameters
            self._initialise_classification_head_kwargs(
                classification_head_kwargs, constructor_args
            )

            self.classifier_head = head_class(
                n_classes=n_classes,
                **classification_head_kwargs
            )
        else:
            self.classifier_head = nn.Linear(
                self.backbone.embed_dim,
                n_classes
            )

        if input_channel_names is not None:
            if isinstance(input_channel_names[0], str):
                channels = [0]  # for cls token
                channels.extend([
                    STANDARD_1020.index(ch) for ch in input_channel_names
                ])
                self.input_chans = channels
            else:
                channels = [0]
                channels.extend(input_channel_names)
                self.input_chans = channels
        else:
            self.input_chans = list(range(input_channels+1))

        trunc_normal_(self.classifier_head.weight, std=0.02)

        self.classifier_head.weight.data.mul_(init_scale)
        self.classifier_head.bias.data.mul_(init_scale)

    def forward(
            self, x: torch.Tensor
        ) -> torch.Tensor:
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
        if self.input_sampling_rate != self.patch_size:
            # Resample the input to match the required sampling rate
            num_samples = int(x.shape[2] * self.in_dim / self.input_sampling_rate)
            x = resample(x, num_samples, axis=2)

        x = x.view(
            x.shape[0], x.shape[1], -1, self.patch_size
        )

        features = self.backbone(x, input_chans=self.input_chans)
        logits = self.classifier_head(features)

        return logits

    def _initialise_classification_head_kwargs(
            self, classification_head_kwargs, constructor_args
        ):
        if 'input_dim' in constructor_args:
            classification_head_kwargs['input_dim'] = self.backbone.embed_dim
        if 'n_classes' in constructor_args:
            del classification_head_kwargs['n_classes']

    def _load_backbone(self, pretrained_weights_path: str, device: str):
        if not pretrained_weights_path:
            raise ValueError(
                "pretrained_weights_path must be provided when use_pretrained_weights is True."
            )

        if pretrained_weights_path.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                pretrained_weights_path, map_location=device, check_hash=True
            )
        else:
            checkpoint = torch.load(
                pretrained_weights_path, map_location=device,
                weights_only=False
            )

        checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint

        # clean unnecessary keys
        if "head.weight" in checkpoint_model:
            del checkpoint_model["head.weight"]
        if "head.bias" in checkpoint_model:
            del checkpoint_model["head.bias"]
        if "relative_position_index" in checkpoint_model:
            del checkpoint_model["relative_position_index"]

        load_state_dict(
            self.backbone, checkpoint_model,
            ignore_missing=['relative_position_index', 'head.weight', 'head.bias'],
            verbose=False
        )

# Standard 10-20 system channel names (for Scalp EEG ONLY)
STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

import torch
import torch.nn as nn

from .classifier import ClassifierModel


class Model(ClassifierModel):
    """
    A PyTorch model for classifying syllables from ECoG data.
    Architecture: Temporal convolutional layers followed by fully connected layers.

    Attributes
    ----------
    feature_extractor : nn.Sequential
        A sequential container of convolutional layers for feature extraction.
    classifier : nn.Sequential
        A sequential container of linear layers for classification.
    n_classes : int
        Number of output classes (e.g., number of syllables).
    latent_length : int
        Temporal length of the input after passing through the feature extractor.
    """
    def __init__(
            self, input_channels: int,
            input_length: int,
            n_classes: int,
            dropout_rate: float=0.5,
            negative_slope: float=0.01
        ) -> None:
        """
        Parameters
        ----------
        input_channels : int
            Number of input channels (e.g., number of electrodes).
        input_length : int
            Length of the input time series (number of timepoints).
        n_classes : int
            Number of output classes (e.g., number of syllables).
        dropout_rate : float, optional
            Dropout rate for regularisation, by default 0.5.
        negative_slope : float, optional
            Slope of the negative part of the LeakyReLU activation function,
            by default 0.01.
        """
        super(Model, self).__init__(n_classes)

        if input_channels <= 0:
            raise ValueError("Input channels must be a positive integer.")

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 512, kernel_size=(3, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=None, padding=0),
            nn.Conv2d(512, 512, kernel_size=(3, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=None, padding=0),
            nn.Conv2d(512, 512, kernel_size=(3, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=None, padding=0),
            nn.Conv2d(512, 512, kernel_size=(3, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=None, padding=0),
            nn.Conv2d(512, 512, kernel_size=(3, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            # nn.MaxPool2d(kernel_size=(2, 1), stride=None, padding=0),
            nn.Conv2d(512, 256, kernel_size=(3, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=None, padding=0),
            nn.Dropout(dropout_rate)
        )

        self.latent_length = self._calculate_temporal_length(input_length)

        if self.latent_length <= 0:
            raise ValueError(
                "Input length is too small for the convolutional layers. "
                "Please increase the input length or adjust the model architecture."
            )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Adjust input size based on the output of conv layers
            nn.Linear(256 * input_channels * self.latent_length, 1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(1024, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_channels, n_timepoints).
            The last dimension is added for compatibility with Conv2d.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_classes) containing the
            predicted probabilities for each class.
        """
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)  # (batch_size, 1, n_timepoints, n_channels)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def _calculate_temporal_length(self, n_timepoints: int) -> int:
        """
        Calculate the temporal length of the input after passing
        through the feature extractor.

        Parameters
        ----------
        n_timepoints : int
            Initial temporal length (height) of the input.
        feature_extractor : nn.Sequential
            Sequential container of convolutional and pooling layers.

        Returns
        -------
        int
            Temporal length after passing through the feature extractor.
        """
        temporal_length = n_timepoints

        for layer in self.feature_extractor:
            if isinstance(layer, nn.Conv2d):
                padding = layer.padding[0] if isinstance(
                    layer.padding, tuple) else layer.padding
                kernel_size, stride = layer.kernel_size[0], layer.stride[0]
            elif isinstance(layer, nn.MaxPool2d):
                padding = layer.padding[0] if isinstance(
                    layer.padding, tuple) else layer.padding
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0] or layer.kernel_size[0]
            else:
                continue

            temporal_length = (temporal_length + 2 * padding - kernel_size) // stride + 1

        return temporal_length

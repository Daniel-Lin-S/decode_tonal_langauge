"""
These architectures follow the design of tone label generator and syllable
label generator in Yan Liu et al. ,
Decoding and synthesizing tonal language speech from brain activity.
Sci. Adv.9,eadh0478(2023).DOI:10.1126/sciadv.adh0478

CNNClassifier was used for syllable classification,
and CNNRNNClassifier was used for tone classification.
"""

import torch
import torch.nn as nn

from .classifier import ClassifierModel


class CNNClassifier(ClassifierModel):
    """
    A PyTorch model for classification of neural recordings.
    Architecture:
    1. Temporal convolutional layers
    2. Fully connected layers.

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
        super(CNNClassifier, self).__init__(n_classes)

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


class CNNRNNClassifier(ClassifierModel):
    """
    A PyTorch model for classification of neural recordings.

    Architecture:
    1. Parallel LSTM and temporal convolutional layers.
    2. Concatenation
    3. Further temporal convolutional layers.

    Attributes
    ----------
    input_channels : int
        Number of input channels (e.g., number of electrodes).
    input_length : int
        Length of the input time series (number of timepoints).
    n_classes : int
        Number of output classes (e.g., number of tones).
    lstm1 : nn.LSTM
        First LSTM layer for processing the input.
    conv_pool_block1 : nn.Sequential
        Convolutional and pooling block on the original input
    conv_pool_block2 : nn.Sequential
        Convolutional and pooling block on the
        output of the first LSTM layer.
    conv_block3 : nn.Sequential
        Convolutional block for further processing.
    dense_lstm_block : nn.Sequential
        Dense layer followed by an LSTM for further processing.
    lstm2 : nn.LSTM
        Second LSTM layer for processing the output of the dense layer.
    output : nn.Linear
        Final linear head for classification.
    """
    def __init__(
            self,
            input_channels: int,
            input_length: int,
            n_classes: int,
            lstm_dim: int = 800,
            dropout: float = 0.5,
            negative_slope: float = 0.01
        ) -> None:
        """
        Parameters
        ----------
        input_channels : int
            Number of input channels (i.e., number of electrodes).
        input_length : int
            Length of the input time series (number of timepoints).
        n_classes : int
            Number of output classes (e.g., number of tones).
        lstm_dim : int, optional
            Dimension of the LSTM hidden state, by default 800.
            It should be divisble by input_length
        dropout : float, optional
            Dropout rate for regularisation, by default 0.5.
        negative_slope : float, optional
            Slope of the negative part of the LeakyReLU activation function,
            by default 0.01.
        """
        super(CNNRNNClassifier, self).__init__(n_classes)

        if lstm_dim % input_length != 0:
            raise ValueError(
                f"lstm_dim ({lstm_dim}) must be divisible "
                f"by input_length ({input_length})."
            )

        self.input_channels = input_channels
        self.input_length = input_length

        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_channels, hidden_size=lstm_dim, batch_first=True)

        # Convolutional and pooling layers
        self.conv_pool_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=1024,
                kernel_size=(7, 1), stride=1, padding=(0, 0)),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        self.conv_pool_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=1024,
                kernel_size=(7, 1), stride=1, padding=(0, 0)),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512,
                      kernel_size=(7, 1), stride=1, padding=(0, 0)),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=512, out_channels=256,
                      kernel_size=(7, 1), stride=1, padding=(0, 0)),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Dropout(dropout)
        )

        w = (lstm_dim // input_length) + input_channels

        self.lstm2 = nn.LSTM(
            input_size=256 * w, hidden_size=512, batch_first=True)

        # Output layer
        self.output = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, T)
            B - batch size, C - number of channels,
            T - number of timepoints.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, n_classes),
            these are the predicted class probabilities.
        """
        batch_size, n_channels, n_timepoints = x.shape

        if n_channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {n_channels}.")

        if n_timepoints != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, got {n_timepoints}.")

        x = x.permute(0, 2, 1)  # (B, T, C)
        x1, _ = self.lstm1(x)  # (B, T, 800)
        x1 = x1[:, -1, :]  # (B, 800)
        
        # parallel convolution blocks
        x = x.view(batch_size, 1, n_timepoints, n_channels)
        x = self.conv_pool_block1(x)  # (B, 1024, t, C)

        x1 = x1.view(batch_size, 1, n_timepoints, -1)  # (B, 1, T, 800//T)
        x1 = self.conv_pool_block2(x1)  # (B, 1024, t, 800//T)

        xf = torch.cat((x1, x), dim=3)

        # combined convolutional block
        x = self.conv_block3(xf)    # (B, 256, t', 800//T+C)
        x = x.view(batch_size, x.shape[2], -1)  # (B, t', 256*(800//T+C))
        x, _ = self.lstm2(x)  # (B, t', 512)

        x = x[:, -1, :]  # (B, 512)
        x = torch.sigmoid(self.output(x))  # (B, n_classes)

        return x
    
    def _compute_temporal_length(self, input_length: int) -> int:
        """
        Compute the length of input after passing through the convolutional layers.
        """
        temporal_length = input_length

        for layer in self.conv_pool_block1:
            if isinstance(layer, nn.Conv2d):
                temporal_length = (
                    temporal_length - layer.kernel_size[0] + 2 * layer.padding[0]
                    ) // layer.stride[0] + 1
            elif isinstance(layer, nn.MaxPool2d):
                temporal_length = (
                    temporal_length - layer.kernel_size[0]) // layer.stride[0] + 1

        # block2 yields the same temporal length as block1

        for layer in self.conv_block3:
            if isinstance(layer, nn.Conv2d):
                temporal_length = (
                    temporal_length - layer.kernel_size[0] + 2 * layer.padding[0]
                    ) // layer.stride[0] + 1
            elif isinstance(layer, nn.MaxPool2d):
                temporal_length = (
                    temporal_length - layer.kernel_size[0]) // layer.stride[0] + 1

        return temporal_length

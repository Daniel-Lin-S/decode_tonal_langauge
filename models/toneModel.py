import torch
from torch import nn

from typing import Dict


class Model(nn.Module):
    """
    A PyTorch model for classifying tones based on ECoG data.

    Attributes
    ----------
    n_channels : int
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
            n_channels: int,
            input_length: int,
            n_classes: int,
            lstm_dim: int = 800,
            dropout: float = 0.5,
            negative_slope: float = 0.01
        ) -> None:
        """
        Parameters
        ----------
        n_channels : int
            Number of input channels (e.g., number of electrodes).
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
        super(Model, self).__init__()

        if lstm_dim % input_length != 0:
            raise ValueError(
                f"lstm_dim ({lstm_dim}) must be divisible "
                f"by input_length ({input_length})."
            )

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.input_length = input_length

        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=n_channels, hidden_size=lstm_dim, batch_first=True)

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

        w = (lstm_dim // input_length) + n_channels

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

        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {n_channels}.")

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

    def get_layer_nparams(self) -> Dict[str, int]:
        """
        Get the number of trainable parameters for
        each layer in the model.
        """
        layer_nparams = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                layer_name = name.split('.')[0]  # Get the layer name
                if layer_name not in layer_nparams:
                    layer_nparams[layer_name] = 0
                layer_nparams[layer_name] += param.numel()
        return layer_nparams

import torch
from torch import nn

from abc import ABC, abstractmethod


class SynthesisModel(nn.Module, ABC):
    """
    Abstract base class for a synthesis model that can be trained
    using the SynthesisTrainer.
    """

    @abstractmethod
    def forward(
        self, inputs_non: torch.Tensor,
        inputs_label: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs_non : torch.Tensor
            Input tensor for non-linguistic features
            with shape (batch_size, input_size).
        inputs_label : torch.Tensor
            Input tensor for tone and syllable dynamics
            with shape (batch_size, label_size).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, output_size).
            The output_size can be variable depending on the target.
            e.g. number of coefficients in the Mel spectrogram.
        """
        pass


class SynthesisModelCNN(SynthesisModel):
    def __init__(
            self,
            output_dim: int,
            n_channels: int,
            n_timepoints: int=200,
            lstm_channels: int=6,
            conv_channels: int=64,
            dropout: float=0.5,
            negative_slope: float=0.01,
        ):
        """
        Parameters
        ----------
        output_dim : int
            Dimension of the output layer.
        n_channels : int
            Number of channels in the ECoG input.
        n_timepoints : int
            Number of time points in the ECoG input.
        lstm_channels : int
            Number of channels in the LSTM output.
        conv_channels : int
            Number of channels in the convolutional layers.
        dropout : float
            Dropout rate for the model.
        negative_slope : float
            Slope of negative part (x < 0) for LeakyReLU activation functions.
        """
        super(SynthesisModelCNN, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.conv_channels = conv_channels
        self.lstm_channels =  lstm_channels
        
        # Input 1: Ec_recon (shape: [batch_size, n_channels, 200])
        self.ecog_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(3, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(in_channels=256, out_channels=conv_channels, kernel_size=(1, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        
        self.ecog_dropout = nn.Dropout(dropout)
        
        # Input 2: labels (shape: [batch_size, 2, n_dynamics])
        self.latent_len = self._compute_latent_length(n_timepoints)
        lstm_size = self.latent_len * n_channels * lstm_channels
        self.label_lstm = nn.LSTM(input_size=2, hidden_size=lstm_size, batch_first=True)
        
        # Concatenation and further processing
        total_channels = conv_channels + lstm_channels
        self.concat_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=total_channels, out_channels=128, kernel_size=(1, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Conv2d(in_channels=128, out_channels=conv_channels, kernel_size=(1, 1), stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1)
        )
        
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(
            conv_channels * self.latent_len * n_channels, output_dim)

    def forward(
            self, inputs_ecog: torch.Tensor, inputs_labels: torch.Tensor
            ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs_ecog : torch.Tensor
            Input tensor with ECoG signals with shape
            (batch_size, n_channels, n_timepoints).
        inputs_labels : torch.Tensor
            A mixture of tone and syllable labels corresponding to the ECoG signals,
            shape (batch_size, 2, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the model.
            Shape (batch_size, output_dim).
        """
        # ecog processing
        x = inputs_ecog.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)   # (batch_size, 1, n_timepoints, n_channels)
        x = self.ecog_conv_block(x)
        x = self.ecog_dropout(x)
        x = x.view(x.size(0), self.conv_channels, self.latent_len, self.n_channels)
        
        # Syllable and tone label processing
        x2 = inputs_labels.permute(0, 2, 1)   # (batch_size, n_dynamics, 2)
        x2, _ = self.label_lstm(x2)  # (batch_size, n_dynamics, lstm_size)
        x2 = x2[:, -1, :]
        x2 = x2.view(x2.size(0), self.lstm_channels, self.latent_len, self.n_channels)
        
        # Concatenate along channel dimension
        x = torch.cat((x, x2), dim=1)

        # Further processing
        x = self.concat_conv_block(x)
        x = self.flatten(x)
        output = self.output_layer(x)
        return output
    
    def _compute_latent_length(self, n_timepoints : int) -> int:
        """
        Compute the latent length after passing
        through the convolutional layers.
        """
        temporal_length = n_timepoints

        for layer in self.ecog_conv_block:
            if isinstance(layer, nn.Conv2d):
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0]
                padding = layer.padding[0]
                temporal_length = (
                    temporal_length - kernel_size + 2 * padding) // stride + 1
            elif isinstance(layer, nn.MaxPool2d):
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0]
                temporal_length = (
                    temporal_length - kernel_size) // stride + 1
        
        return temporal_length


class SynthesisLite(SynthesisModel):
    def __init__(
            self,
            output_dim: int,
            n_channels: int,
            n_timepoints: int = 200,
            label_dim: int = 2,
            conv_channels: int = 32,
            lstm_hidden: int = 64,
            dropout: float = 0.3,
            negative_slope: float = 0.01
        ) -> None:
        """
        Parameters
        ----------
        output_dim : int
            Dimension of the output layer.
        n_channels : int
            Number of channels in the ECoG input.
        n_timepoints : int
            Number of time points in the ECoG input.
        label_dim : int
            Dimension of the label sequence input (e.g., syllable and tone).
        conv_channels : int
            Number of channels in the convolutional layers.
        lstm_hidden : int
            Hidden size of the LSTM layer.
        dropout : float
            Dropout rate for the model.
        negative_slope : float
            Slope of negative part (x < 0) for LeakyReLU activation functions.
        """
        super(SynthesisLite, self).__init__()

        # ECoG block: depthwise-separable Conv1D + 2 poolings
        self.ecog_conv = nn.Sequential(
            nn.Conv1d(n_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool1d(kernel_size=2)
        )

        ecog_feature_len = n_timepoints // 4
        self.ecog_out_dim = conv_channels * ecog_feature_len

        # Label sequence block: LSTM over (syllable, tone)
        self.label_lstm = nn.LSTM(input_size=label_dim,
                                  hidden_size=lstm_hidden,
                                  batch_first=True,
                                  bidirectional=False)

        # Final projection (FC layers)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.ecog_out_dim + lstm_hidden, 512),
            nn.LeakyReLU(negative_slope),
            nn.Linear(512, output_dim)
        )

    def forward(
            self, x_ecog: torch.Tensor, x_label: torch.Tensor
        ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_ecog : torch.Tensor
            ECoG input tensor with shape (B, C, T),
            where B is batch size, C is number of channels,
            and T is number of time points.
        x_label : torch.Tensor
            Label sequence tensor with shape (B, 2, L),
            where B is batch size, 2 corresponds to syllable and tone,
            and L is the length of the label sequence.

        Returns
        -------
        Tensor
            with shape (B, output_dim), the output of the model.
        """
        # ECoG: (B, C, T) → (B, conv_channels, T') → flatten
        x_ecog_feat = self.ecog_conv(x_ecog)
        x_ecog_feat = x_ecog_feat.flatten(1)

        # Labels: (B, 2, L) → (B, L, 2)
        x_label = x_label.permute(0, 2, 1)
        _, (h_n, _) = self.label_lstm(x_label)  # h_n: (1, B, H)
        x_label_feat = h_n.squeeze(0)

        # Concatenate + MLP
        x = torch.cat([x_ecog_feat, x_label_feat], dim=-1)
        return self.fc(x)

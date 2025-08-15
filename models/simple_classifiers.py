import torch.nn as nn
import torch
from typing import Optional

from .classifier import ClassifierModel
from .utils import get_activation


class LogisticRegressionClassifier(ClassifierModel):
    """
    A simple logistic regression classifier as a benchmark model.

    Attributes
    ----------
    linear : nn.Linear
        Linear layer for classification.
    input_dim : int
        Number of input features expected by the model.
    """

    def __init__(self, input_dim: int, n_classes: int):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features.
        n_classes : int
            Number of classes for classification.
        """
        super(LogisticRegressionClassifier, self).__init__(n_classes)
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the logistic regression model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, ...).
            The last dimension is flattened if it has more than 2 dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, n_classes).
            Each entry represents the predicted logits for each class.
        """
        if x.ndim > 2:
            # Flatten the input tensor
            x = x.view(x.size(0), -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {x.shape[1]}."
            )

        return self.linear(x)


class ShallowNNClassifier(ClassifierModel):
    """
    A shallow feedforward neural network classifier as a benchmark model.
    
    Attributes
    ----------
    input_dim : int
        Number of input features expected by the model.
    hidden : nn.Linear
        Linear layer for the hidden layer.
    output : nn.Linear
        Linear layer for the output layer.
    activation : nn.ReLU
        Activation function applied to the hidden layer.
    """

    def __init__(
            self, input_dim: int,
            n_classes: int,
            hidden_dim: Optional[int]=None,
            activation: str='ReLU'
        ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Number of input features.
        n_classes : int
            Number of classes for classification.
        hidden_dim : int, optional
            Number of hidden units in the hidden layer.
            If None, it defaults to half of the input dimension.
        """
        super(ShallowNNClassifier, self).__init__(n_classes)

        self.input_dim = input_dim

        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_classes)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the shallow neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, ...)
            the last dimension is flattened if it has more than 2 dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, n_classes).
            Each entry represents the predicted logits for each class.
        """
        if x.ndim > 2:
            # Flatten the input tensor
            x = x.view(x.size(0), -1)
        
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {x.shape[1]}."
            )

        x = self.activation(self.hidden(x))
        return self.output(x)

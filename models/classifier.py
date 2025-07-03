from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class ClassifierModel(nn.Module, ABC):
    """
    Abstract base class for a classifier model that can be trained
    using the ClassifierTrainer.

    Attributes
    ----------
    n_classes : int
        Number of classes for classification.
    """

    def __init__(self, n_classes: int):
        """
        Parameters
        ----------
        n_classes : int
            Number of classes for classification.
        """
        super(ClassifierModel, self).__init__()
        if n_classes < 2:
            raise ValueError("Number of classes must be at least 2.")
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, n_classes).
            Each entry represents the predicted probability
            for each class.
        """
        pass
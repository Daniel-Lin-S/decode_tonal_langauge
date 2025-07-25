from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from typing import Dict


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

    def get_layer_nparams(self) -> Dict[str, int]:
        """
        Get the number of trainable parameters for
        each layer in the model.

        Returns
        -------
        Dict[str, int]
            Dictionary where keys are layer names and values
            are the number of trainable parameters in that layer.
        """
        layer_nparams = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                layer_name = name.split('.')[0]  # Get the layer name
                if layer_name not in layer_nparams:
                    layer_nparams[layer_name] = 0
                layer_nparams[layer_name] += param.numel()
        return layer_nparams

    def get_nparams(self) -> int:
        """
        Get the total number of trainable parameters in the model.
        
        Returns
        -------
        int
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import NAdam
import numpy as np
from typing import Tuple, List


def compute_mcd(true_mcc: torch.Tensor, pred_mcc: torch.Tensor) -> float:
    """
    Compute Mel-Cepstral Distortion (MCD) between
    two sequences of Mel-Cepstral Coefficients.
    Average MCD is computed over the batch.

    Parameters
    ----------
    true_mcc : torch.Tensor
        The true Mel-Cepstral Coefficients (in dB) with shape
        (batch_size, n_coeffs).
    pred_mcc : torch.Tensor
        The predicted Mel-Cepstral Coefficients (in dB) with shape
        (batch_size, n_coeffs).

    Returns
    -------
    mcd : float
        The computed Mel-Cepstral Distortion.
    """
    # Ensure inputs are tensors
    true_mcc = true_mcc.float()
    pred_mcc = pred_mcc.float()

    squared_diff = torch.sum((true_mcc - pred_mcc) ** 2, dim=1)

    # MCD formula
    mcd = torch.mean(10 / np.log(10) * torch.sqrt(2 * squared_diff))

    return mcd.item()


class SynthesisTrainer:
    """
    A trainer for PyTorch models used for classification tasks.
    It uses the NAdam optimizer and
    supports both binary and multi-class classification.

    Attributes
    ----------
    model : torch.nn.Module
        The model to be trained.
    device : torch.device
        The device on which the model will be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    criterion : torch.nn.Module
        The loss function used for training.
    metric : torchmetrics.Metric
        The metric used to evaluate the model's performance.
        MulticlassAccuracy used in this model.
    """

    def __init__(
            self, model: torch.nn.Module,
            device: torch.device = torch.device("cpu"),
            learning_rate: float=0.0005,
            beta_1: float=0.9,
            beta_2: float=0.999,
            epsilon: float=1e-08,
            schedule_decay: float=0.004,
            verbose: bool=True
        ) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
        device : torch.device, optional
            The device on which the model will be trained,
            by default torch.device("cpu").
        n_classes : int
            Number of classes for classification.
        learning_rate : float, optional
            Learning rate for the optimizer, by default 0.0005.
        beta_1 : float, optional
            The exponential decay rate for the first moment estimates,
            by default 0.9.
        beta_2 : float, optional
            The exponential decay rate for the second moment estimates,
            by default 0.999.
        epsilon : float, optional
            Term added to the denominator to improve numerical stability,
            by default 1e-08.
        schedule_decay : float, optional
            Weight decay for the optimizer, by default 0.004.
        verbose : bool, optional
            If True, prints the number of trainable parameters,
            by default True.
        """
        self.device = device
        self.model = model.to(device)

        if verbose:
            n_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            print(f"Number of trainable parameters: {n_trainable_params:,}")

        self.optimizer = NAdam(
            model.parameters(),
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=schedule_decay
        )

        # Mean absolute error 
        self.criterion = nn.L1Loss()


    def train(
            self, train_loader: DataLoader,
            epochs: int,
            verbose: bool=True
        ) -> List[Tuple[float, float]]:
        """
        Train the model using the provided training data loader.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training dataset.
        epochs : int
            Number of epochs to train the model.
        verbose : bool, optional
            If True, prints training progress, by default True.

        Return
        -------
        history : list of tuples
            A list containing tuples of (epoch_loss, epoch_accuracy)
            for each epoch.
        """
        self.model.train()
        history = []
        for epoch in range(epochs):
            epoch_loss = 0
            mcd = 0
            for inputs_ecog, inputs_labels, targets in train_loader:
                # forward pass
                inputs_ecog = inputs_ecog.to(self.device)
                inputs_labels = inputs_labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs_ecog, inputs_labels)
                targets = targets.long().to(self.device)

                # loss and metric computations
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                mcd += compute_mcd(targets, outputs)

            epoch_loss /= len(train_loader)
            mcd /= len(train_loader)
            history.append((epoch_loss, mcd))
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, "
                      f"Mean MCD: {mcd:.4f}")

        return history

    def evaluate(
            self, test_loader: DataLoader
        ) -> Tuple[torch.Tensor, float]:
        """
        Evaluate the model on the test dataset.

        Parameters
        ----------
        test_loader : DataLoader
            DataLoader for the test dataset.

        Returns
        -------
        mcd : float
            Mean Mel-Cepstral Distortion (MCD) of
            the model on the test dataset.
        recon_mels : numpy.ndarray
            The predicted Mel-Cepstral Coefficients
            for the test dataset.
        origin_mels : numpy.ndarray
            The original Mel-Cepstral Coefficients
            for the test dataset.
        """
        self.model.eval()

        recon_mels = []
        origin_mels = []

        with torch.no_grad():
            mcd = 0
            for inputs_ecog, inputs_labels, targets in test_loader:
                inputs_labels = inputs_labels.to(self.device)
                inputs_ecog = inputs_ecog.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs_ecog, inputs_labels)

                mcd += compute_mcd(targets, outputs)
                recon_mels.append(outputs.cpu())
                origin_mels.append(targets.cpu())
            
            mcd /= len(test_loader)

        recon_mels = torch.cat(recon_mels, dim=0)
        origin_mels = torch.cat(origin_mels, dim=0)

        return mcd, recon_mels.numpy(), origin_mels.numpy()

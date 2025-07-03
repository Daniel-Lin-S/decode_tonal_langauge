import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import NAdam
import numpy as np
from typing import Tuple, List, Dict

from data_loading.utils import prepare_tone_dynamics


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
            self,
            synthesize_model: torch.nn.Module,
            tone_model: torch.nn.Module,
            syllable_model: torch.nn.Module,
            tone_dynamic_mapping: Dict[str, List[int]],
            device: torch.device = torch.device("cpu"),
            learning_rate: float=0.0005,
            beta_1: float=0.9,
            beta_2: float=0.999,
            epsilon: float=1e-08,
            schedule_decay: float=0.004,
            verbose: bool=True,
            train_classifiers: bool=False
        ) -> None:
        """
        Parameters
        ----------
        synthesize_model : torch.nn.Module
            The speech synthesis model to be trained.
        tone_model : torch.nn.Module
            The tone classification model.
        syllable_model : torch.nn.Module
            The syllable classification model.
        tone_dynamic_mapping : dict
            A dictionary mapping tone indices to their
            corresponding dynamics.
            e.g. "2" : [1, 2,  3, 2, 1]
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
        train_classifiers : bool, optional
            If True, the tone and syllable classification models
            will be trained along with the synthesis model.
        """
        self.device = device
        self.train_classifiers = train_classifiers
        self.tone_dynamic_mapping = tone_dynamic_mapping
        self.model = synthesize_model.to(device)

        if verbose:
            n_trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            print(
                "Number of trainable parameters in the synthesis model: "
                f"{n_trainable_params:,}")

        self.optimizer = NAdam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=schedule_decay
        )

        # Mean absolute error 
        self.criterion = nn.L1Loss()

        self.tone_model = tone_model
        self.tone_model.to(device)
        self.syllable_model = syllable_model
        self.syllable_model.to(device)

        if not train_classifiers:
            self.tone_model.eval()
            self.syllable_model.eval()


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

        if self.train_classifiers:
            self.tone_model.train()
            self.syllable_model.train()

        history = []
        for epoch in range(epochs):
            epoch_loss = 0
            mcd = 0
            for inputs_non, inputs_syllable, inputs_tone, targets in train_loader:
                # forward pass
                inputs_non = inputs_non.to(self.device)
                inputs_syllable = inputs_syllable.to(self.device)
                inputs_tone = inputs_tone.to(self.device)

                tone_labels = torch.argmax(
                    self.tone_model(inputs_tone), dim=1)
                syllable_labels = torch.argmax(
                    self.syllable_model(inputs_syllable), dim=1)

                inputs_label = prepare_tone_dynamics(
                    self.tone_dynamic_mapping,
                    tone_labels.cpu().numpy(),
                    syllable_labels.cpu().numpy()
                )

                inputs_label = torch.Tensor(inputs_label).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs_non, inputs_label)
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
        self.tone_model.eval()
        self.syllable_model.eval()

        recon_mels = []
        origin_mels = []

        with torch.no_grad():
            mcd = 0
            for inputs_non, inputs_syllable, inputs_tone, targets in test_loader:
                inputs_non = inputs_non.to(self.device)
                inputs_syllable = inputs_syllable.to(self.device)
                inputs_tone = inputs_tone.to(self.device)

                # take the label with highest probability
                tone_labels = torch.argmax(
                    self.tone_model(inputs_tone), dim=1)
                syllable_labels = torch.argmax(
                    self.syllable_model(inputs_syllable), dim=1)

                inputs_label = prepare_tone_dynamics(
                    self.tone_dynamic_mapping,
                    tone_labels.cpu().numpy(),
                    syllable_labels.cpu().numpy()
                )

                inputs_label = torch.Tensor(inputs_label).to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs_non, inputs_label)

                mcd += compute_mcd(targets, outputs)
                recon_mels.append(outputs.cpu())
                origin_mels.append(targets.cpu())
            
            mcd /= len(test_loader)

        recon_mels = torch.cat(recon_mels, dim=0)
        origin_mels = torch.cat(origin_mels, dim=0)

        return mcd, recon_mels.numpy(), origin_mels.numpy()

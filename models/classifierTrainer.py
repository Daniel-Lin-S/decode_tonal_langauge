import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import NAdam
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
)

from typing import Tuple, List, Optional, Dict

from .classifier import ClassifierModel


class ClassifierTrainer:
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
            self, model: ClassifierModel,
            device: torch.device = torch.device("cpu"),
            n_classes: Optional[int]=None,
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

        if hasattr(model, 'n_classes'):
            n_classes = model.n_classes
        elif n_classes is None:
            raise ValueError(
                "Number of classes must be specified "
                "or the model must have n_classes attribute."
            )
        
        if n_classes < 2:
            raise ValueError(
                "Number of classes must be at least 2."
            )
    
        self.n_classes = n_classes

        self.criterion = nn.CrossEntropyLoss()
        self.acc_metric = MulticlassAccuracy(
            num_classes=n_classes,
            average='macro'
        ).to(device)
        self.f1_metric = MulticlassF1Score(
            num_classes=n_classes,
            average='macro'
        ).to(device)
        self.confusion_metric = MulticlassConfusionMatrix(
            num_classes=n_classes
        ).to(device)


    def train(
            self, train_loader: DataLoader,
            epochs: int,
            vali_loader: Optional[DataLoader]=None,
            patience: int=5,
            verbose: bool=True
        ) -> List[Dict[str, float]]:
        """
        Train the model using the provided training data loader.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training dataset.
        epochs : int
            Number of epochs to train the model.
        vali_loader : DataLoader, optional
            DataLoader for the validation dataset.
            If provided, early stopping will be
            applied based on validation loss,
            by default None.
        patience : int, optional
            Number of epochs with no improvement
            after which training will be stopped.
            Used for early stopping, by default 5.
        verbose : bool, optional
            If True, prints training progress, by default True.

        Return
        -------
        history : list of dicts
            A list of dictionaries containing training and validation
            loss and accuracy for each epoch.
            Keys: 'train_loss', 'train_accuracy', and
            'vali_loss', 'vali_accuracy' (only if vali_loader is provided).
        """
        self.model.train()
        history = []
        best_vali_loss = float('inf')
        counter = 0   # Counter for early stopping

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                targets = targets.long().to(self.device)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_accuracy = self.acc_metric(outputs, targets).item()

            epoch_loss /= len(train_loader)
            epoch_accuracy /= len(train_loader)

            if vali_loader is not None:
                vali_loss, vali_accuracy = self.vali(vali_loader)

                if vali_loss < best_vali_loss:
                    best_vali_loss = vali_loss
                    if verbose:
                        print(
                            "Validaton loss improved, saving model..."
                        )
                    best_model_states = self.model.state_dict()
                else:
                    counter += 1

                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, "
                          f"Train Loss: {epoch_loss:.4f}, "
                          f"Train Accuracy: {epoch_accuracy:.4f}, "
                          f"Validation Loss: {vali_loss:.4f}"
                          f", Validation Accuracy: {vali_accuracy:.4f}"
                        )

                history.append({
                    'train_loss': epoch_loss,
                    'train_accuracy': epoch_accuracy,
                    'vali_loss': vali_loss,
                    'vali_accuracy': vali_accuracy
                })

                if counter >= patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch+1}, "
                            "no improvement in validation loss."
                        )
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, "
                        f"Training Loss: {epoch_loss:.4f}"
                        f", Training Accuracy: {epoch_accuracy:.4f}")
                
                history.append({
                    'train_loss': epoch_loss,
                    'train_accuracy': epoch_accuracy
                })

        if vali_loader is not None:
            self.model.load_state_dict(best_model_states)

        return history
    
    def vali(self, vali_loader: DataLoader) -> torch.Tensor:
        """
        Validate the model using the provided validation data loader.

        Parameters
        ----------
        vali_loader : DataLoader
            DataLoader for the validation dataset.

        Returns
        -------
        vali_loss : float
            The validation loss computed over the validation dataset,
            averaged across all batches.
        vali_accuracy : float
            The validation accuracy computed over the validation dataset,
            averaged across all batches.
        """
        self.model.eval()
        self.acc_metric.reset()

        vali_loss = 0.0
        vali_accuracy = 0.0

        with torch.no_grad():
            for inputs, targets in vali_loader:
                inputs = inputs.to(self.device)
                targets = targets.long().to(self.device)
                outputs = self.model(inputs)

                vali_loss += self.criterion(outputs, targets).item()
                vali_accuracy += self.acc_metric(outputs, targets).item()

        vali_loss /= len(vali_loader)
        vali_accuracy /= len(vali_loader)

        self.model.train()
        self.acc_metric.reset()

        return vali_loss, vali_accuracy

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
        accuracy : float
            Mean accuracy of the model on the test dataset
            across all batches.
        f1_score : float
            Mean F1 score of the model on the test dataset
            across all batches.
        confusion_matrix : np.ndarray
            Confusion matrix of the model's predictions on the test dataset.
        """
        self.model.eval()

        self.acc_metric.reset()
        self.f1_metric.reset()
        self.confusion_metric.reset()

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                labels_pred = outputs.argmax(dim=1)

                self.acc_metric.update(outputs, targets)
                self.f1_metric.update(outputs, targets)
                self.confusion_metric.update(labels_pred, targets)

        accuracy = self.acc_metric.compute().item()
        f1_score = self.f1_metric.compute().item()
        confusion_matrix = self.confusion_metric.compute().cpu().numpy()

        return accuracy, f1_score, confusion_matrix

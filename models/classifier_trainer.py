"""Utilities for training classifiers with PyTorch Lightning."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import NAdam
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix
)
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger, CSVLogger
import os
import pandas as pd


from .classifier import ClassifierModel


class LightningClassifier(pl.LightningModule):
    """Lightning module wrapping a :class:`ClassifierModel`."""

    def __init__(
            self, model: ClassifierModel,
            learning_rate: float=0.0005,
        ) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
        learning_rate : float, optional
            Learning rate for the optimizer, by default 0.0005.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.criterion = nn.CrossEntropyLoss()
        self.acc_metric = MulticlassAccuracy(
            num_classes=model.n_classes,
            average='macro'
        )
        self.f1_metric = MulticlassF1Score(
            num_classes=model.n_classes,
            average='macro'
        )
        self.confusion_metric = MulticlassConfusionMatrix(num_classes=model.n_classes)

        # for evaluations
        self.confusion_matrix: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Forward & Optimiser
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)

    def configure_optimizers(self):
        return NAdam(self.model.parameters(), lr=self.learning_rate)

    # ------------------------------------------------------------------
    # Training / Validation Steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        y = y.long()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        accuracy = self.acc_metric(logits, y)

        self.log("train/loss", loss, prog_bar=False,
                 on_step=True, on_epoch=True, logger=True)
        self.log("train/accuracy", accuracy, prog_bar=False,
                    on_step=False, on_epoch=True, logger=True)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        y = y.long()
        logits = self.model(x)

        loss = self.criterion(logits, y)
        self.log("val/loss", loss, prog_bar=False,
            on_epoch=True, logger=True)

        accuracy = self.acc_metric(logits, y)
        self.log("val/accuracy", accuracy, prog_bar=False,
                 on_epoch=True, logger=True)

        return {"val/loss": loss}

    # ------------------------------------------------------------------
    # Testing / Prediction
    # ------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self.model(x)
        preds = logits.argmax(dim=1)

        self.acc_metric.update(logits, y)
        self.f1_metric.update(logits, y)
        self.confusion_metric.update(preds, y)

    def on_test_epoch_end(self) -> None:
        self.test_accuracy = self.acc_metric.compute().item()
        self.test_f1 = self.f1_metric.compute().item()
        self.confusion_matrix = self.confusion_metric.compute()

        csv_log_dir = self._get_logger_dir(CSVLogger)

        cm_path = os.path.join(csv_log_dir, "confusion_matrix_test.csv")
        cm_array = self.confusion_matrix.cpu().numpy()
        pd.DataFrame(cm_array).to_csv(cm_path, index=False, header=False)

        self.acc_metric.reset()
        self.f1_metric.reset()
        self.confusion_metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):  # type: ignore[override]
        x, _ = batch
        logits = self.model(x)
        return logits.argmax(dim=1)

    # ------------------------------------------------------------------
    # Helper proxies to underlying model
    # ------------------------------------------------------------------
    def get_nparams(self) -> int:
        return self.model.get_nparams()

    def get_layer_nparams(self) -> Dict[str, int]:
        return self.model.get_layer_nparams()

    def _get_logger_dir(self, logger_class: Optional[Logger]=None) -> str:
        """
        Get the directory where the CSV logger is saving logs.
        This is useful for saving additional metrics or results.
        """
        if self.logger is None:
            raise ValueError("No logger is set for this model.")

        if isinstance(self.logger, logger_class):
            return self.logger.log_dir
        elif isinstance(self.logger, list):
            for logger in self.logger:
                if isinstance(logger, logger_class):
                    return logger.log_dir
            raise ValueError(f"No logger of type {logger_class} found in the list.")

        return self.logger.log_dir

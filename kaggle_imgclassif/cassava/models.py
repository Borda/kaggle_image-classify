from typing import Union

import timm
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score


class LitCassava(LightningModule):
    """Basic Cassava model.

    >>> model = LitCassava("resnet18")
    """

    def __init__(self, model: Union[str, nn.Module], num_classes: int = 5, lr: float = 1e-4):
        super().__init__()
        if isinstance(model, str):
            self.model = timm.create_model(model, pretrained=True, num_classes=num_classes)
        else:
            self.model = model
        self.accuracy = Accuracy()
        self.f1_score = F1Score(num_classes)
        self.learn_rate = lr
        self.loss_fn = F.cross_entropy

    def forward(self, x):
        return F.softmax(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("valid_loss", loss, prog_bar=False)
        self.log("valid_acc", self.accuracy(y_hat, y), prog_bar=True)
        self.log("valid_f1", self.f1_score(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learn_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]

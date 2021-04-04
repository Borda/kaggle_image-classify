import torch
import torchmetrics
import torchvision
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F


class LitResnet(nn.Module):

    def __init__(self, arch: str, pretrained: bool = True, num_classes: int = 6):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.model = torchvision.models.__dict__[arch](pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class LitPlantPathology(LightningModule):
    """
    This model is meant and tested to be used together with `PlantPathologySimpleDataset`
    """

    def __init__(self, model, lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.arch = self.model.arch
        self.num_classes = self.model.num_classes
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_f1_score = torchmetrics.F1(self.num_classes)
        self.learn_rate = lr
        self.loss_fn = F.cross_entropy

    def on_epoch_start(self):
        if self.trainer.current_epoch < 2:
            return
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        # print("UNFREEZE")

    def forward(self, x):
        return F.softmax(self.model(x))

    def compute_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy(y_hat, y), prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log("valid_loss", loss, prog_bar=False)
        self.log("valid_acc", self.val_accuracy(y_hat, y), prog_bar=True)
        self.log("valid_f1", self.val_f1_score(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learn_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]


class MultiPlantPathology(LitPlantPathology):
    """
    This model is meant and tested to be used together with `PlantPathologyDataset`
    """

    def __init__(self, model, lr: float = 1e-4):
        super().__init__(model, lr)
        self.loss = nn.BCEWithLogitsLoss()

    def compute_loss(self, y_hat, y):
        return self.loss(y_hat, y.to(float))

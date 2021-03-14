import torch
import torchmetrics
import torchvision
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F


class LitResnet(nn.Module):

    def __init__(self, arch: str, pretrained: bool = True, num_classes: int = 5):
        super().__init__()
        self.model = torchvision.models.__dict__[arch](pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class LitMobileNet(nn.Module):

    def __init__(self, arch: str, pretrained: bool = True, num_classes: int = 5):
        super().__init__()
        self.model = torchvision.models.__dict__[arch](pretrained=pretrained)
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class LitCassava(LightningModule):

    def __init__(self, model, num_classes: int = 5, lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1(num_classes)
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

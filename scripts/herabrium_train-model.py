import json
import os

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import fire

import flash
import pandas as pd
import torch
from flash.core.data.io.input_transform import InputTransform
from flash.image import ImageClassificationData, ImageClassifier

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import F1Score
from torchvision import transforms as T


@dataclass
class ImageClassificationInputTransform(InputTransform):

    image_size: Tuple[int, int] = (224, 224)
    image_color_mean: Tuple[float, float] = (0.781, 0.759, 0.710)
    image_color_std: Tuple[float, float] = (0.241, 0.245, 0.249)

    def input_per_sample_transform(self):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(self.image_size),
                T.Normalize(self.image_color_mean, self.image_color_std),
            ]
        )

    def train_input_per_sample_transform(self):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(self.image_size),
                T.Normalize(self.image_color_mean, self.image_color_std),
                T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.1, 0.1)),
                # T.ColorJitter(),
                # T.RandomAutocontrast(),
                # T.RandomPerspective(distortion_scale=0.1),
            ]
        )

    def target_per_sample_transform(self) -> Callable:
        return torch.as_tensor


def main(
    dataset_dir: str = "/home/jirka/Datasets/herbarium-2022-fgvc9",
    checkpoints_dir: str = "/home/jirka/Workspace/checkpoints_herbarium-flash",
    batch_size: int = 24,
    num_workers: int = 12,
    model_backbone: str = "efficientnet_b3",
    model_pretrained: bool = False,
    optimizer: str = "AdamW",
    image_size: int = 300,
    lr_scheduler: Optional[str] = None,
    learning_rate: float = 5e-3,
    max_epochs: int = 20,
    gpus: int = 1,
    accumulate_grad_batches: int = 1,
    early_stopping: Optional[float] = None,
    swa: Optional[float] = None,
) -> None:
    with open(os.path.join(dataset_dir, "train_metadata.json")) as fp:
        train_data = json.load(fp)
    train_annotations = pd.DataFrame(train_data["annotations"])
    train_images = pd.DataFrame(train_data["images"]).set_index("image_id")
    train_categories = pd.DataFrame(train_data["categories"]).set_index("category_id")
    train_institutions = pd.DataFrame(train_data["institutions"]).set_index("institution_id")
    df_train = pd.merge(train_annotations, train_images, how="left", right_index=True, left_on="image_id")
    df_train = pd.merge(df_train, train_categories, how="left", right_index=True, left_on="category_id")
    df_train = pd.merge(df_train, train_institutions, how="left", right_index=True, left_on="institution_id")

    datamodule = ImageClassificationData.from_data_frame(
        input_field="file_name",
        target_fields="category_id",
        # for simplicity take just half of the data
        train_data_frame=df_train,
        # train_data_frame=df_train[:len(df_train) // 2],
        train_images_root=os.path.join(dataset_dir, "train_images"),
        train_transform=ImageClassificationInputTransform,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_kwargs={"image_size": (image_size, image_size)},
    )

    model = ImageClassifier(
        backbone=model_backbone,
        metrics=F1Score(num_classes=datamodule.num_classes),
        pretrained=model_pretrained,
        optimizer=optimizer,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        num_classes=datamodule.num_classes,
    )

    # Trainer Args
    logger = WandbLogger(project="Flash_tract-image-segmentation")
    log_id = str(logger.experiment.id)
    monitor = "val_f1score"
    cbs = [ModelCheckpoint(dirpath=checkpoints_dir, filename=f"{log_id}", monitor=monitor, mode="max", verbose=True)]
    if early_stopping is not None:
        cbs.append(EarlyStopping(monitor=monitor, min_delta=early_stopping, mode="max", verbose=True))
    if isinstance(swa, float):
        cbs.append(StochasticWeightAveraging(swa_epoch_start=swa))

    trainer = flash.Trainer(
        callbacks=cbs,
        max_epochs=max_epochs,
        # precision="bf16",
        gpus=gpus,
        accelerator="ddp" if gpus > 1 else None,
        benchmark=True,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # Train the model
    # trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
    trainer.finetune(model, datamodule=datamodule, strategy=("freeze_unfreeze", 5))

    trainer.save_checkpoint("image_classification_model.pt")

    # Save the model!
    checkpoint_name = f"tract-segm-{log_id}_{model_backbone}.pt"
    trainer.save_checkpoint(os.path.join(checkpoints_dir, checkpoint_name))


if __name__ == "__main__":
    fire.Fire(main)

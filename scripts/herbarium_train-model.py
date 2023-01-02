"""Sample execution on A100.

>> python3 scripts/herbarium_train-model.py --gpus 6 --max_epochs 30 --val_split 0.05 \
    --learning_rate 0.01 --model_backbone convnext_base_384_in22ft1k --image_size 384 --model_pretrained True \
    --batch_size 72 --label_smoothing None --accumulate_grad_batches=12

>> python3 scripts/herbarium_train-model.py --gpus 6 --max_epochs 30 --val_split 0.05 \
    --learning_rate 0.001 --model_backbone dm_nfnet_f3 --image_size 416 --model_pretrained True \
    --batch_size 18 --accumulate_grad_batches=48
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import fire
import flash
import pandas as pd
import torch
from flash.core.data.io.input_transform import InputTransform
from flash.image import ImageClassificationData, ImageClassifier
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from timm.loss import AsymmetricLossSingleLabel, LabelSmoothingCrossEntropy
from torchmetrics import F1Score
from torchvision import transforms as T


@dataclass
class ImageClassificationInputTransform(InputTransform):

    image_size: Tuple[int, int] = (224, 224)
    color_mean: Tuple[float, float, float] = (0.781, 0.759, 0.710)
    color_std: Tuple[float, float, float] = (0.241, 0.245, 0.249)

    def input_per_sample_transform(self):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(self.image_size),
                T.Normalize(self.color_mean, self.color_std),
            ]
        )

    def train_input_per_sample_transform(self):
        return T.Compose(
            [
                T.TrivialAugmentWide(),
                T.RandomPosterize(bits=6),
                T.RandomEqualize(),
                T.ToTensor(),
                T.Resize(self.image_size),
                T.RandomHorizontalFlip(),
                # T.ColorJitter(brightness=0.2, hue=0.1),
                T.RandomAutocontrast(),
                T.RandomAdjustSharpness(sharpness_factor=2),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                T.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.1, 0.1)),
                # T.RandomPerspective(distortion_scale=0.1),
                T.Normalize(self.color_mean, self.color_std),
            ]
        )

    def target_per_sample_transform(self) -> Callable:
        return torch.as_tensor


def load_df_train(dataset_dir: str) -> pd.DataFrame:
    with open(os.path.join(dataset_dir, "train_metadata.json")) as fp:
        train_data = json.load(fp)
    train_annotations = pd.DataFrame(train_data["annotations"])
    train_images = pd.DataFrame(train_data["images"]).set_index("image_id")
    train_categories = pd.DataFrame(train_data["categories"]).set_index("category_id")
    train_institutions = pd.DataFrame(train_data["institutions"]).set_index("institution_id")
    df_train = pd.merge(train_annotations, train_images, how="left", right_index=True, left_on="image_id")
    df_train = pd.merge(df_train, train_categories, how="left", right_index=True, left_on="category_id")
    df_train = pd.merge(df_train, train_institutions, how="left", right_index=True, left_on="institution_id")
    df_train["file_name"] = df_train["file_name"].apply(lambda p: os.path.join("train_images", p))
    return df_train


def append_predictions(df_train: pd.DataFrame, path_csv: str = None) -> pd.DataFrame:
    if not path_csv:
        return df_train
    if os.path.isfile(path_csv):
        raise FileNotFoundError(f"Missing predictions: {path_csv}")
    df_preds = pd.read_csv(path_csv)
    df_preds["file_name"] = df_preds["file_name"].apply(lambda p: os.path.join("test_images", p))
    df_train.append(df_preds)
    return df_train


def inference(
    model, df_test: pd.DataFrame, dataset_dir: str, image_size: int, batch_size: int, gpus: int = 0
) -> pd.DataFrame:
    print(f"inference for {len(df_test)} images")
    print(df_test.head())

    datamodule = ImageClassificationData.from_data_frame(
        input_field="file_name",
        # target_fields="category_id",
        predict_data_frame=df_test,
        # for simplicity take just fraction of the data
        # predict_data_frame=test_images[:len(test_images) // 100],
        predict_images_root=os.path.join(dataset_dir, "test_images"),
        predict_transform=ImageClassificationInputTransform,
        batch_size=batch_size,
        transform_kwargs={"image_size": (image_size, image_size)},
        num_workers=batch_size,
    )

    trainer = flash.Trainer(gpus=min(gpus, 1))

    predictions = []
    for lbs in trainer.predict(model, datamodule=datamodule, output="labels"):
        # lbs = [torch.argmax(p["preds"].float()).item() for p in preds]
        predictions += lbs

    print(f"Predictions: {len(predictions)} & Test images: {len(df_test)}")
    df_test["category_id"] = predictions
    return df_test


def main(
    dataset_dir: str = "/home/jirka/Datasets/herbarium-2022-fgvc9",
    checkpoints_dir: str = "/home/jirka/Workspace/checkpoints_herbarium-flash",
    predict_csv: str = None,
    model_backbone: str = "efficientnet_b3",
    model_pretrained: bool = False,
    image_size: int = 320,
    optimizer: str = "AdamW",
    lr_scheduler: Optional[str] = None,
    learning_rate: float = 5e-3,
    label_smoothing: float = 0.01,
    batch_size: int = 24,
    max_epochs: int = 20,
    gpus: int = 1,
    val_split: float = 0.1,
    early_stopping: Optional[float] = None,
    swa: Optional[float] = None,
    num_workers: int = None,
    run_inference: bool = True,
    **trainer_kwargs: Dict[str, Any],
) -> None:
    print(f"Additional Trainer args: {trainer_kwargs}")
    df_train = load_df_train(dataset_dir)

    with open(os.path.join(dataset_dir, "test_metadata.json")) as fp:
        test_data = json.load(fp)
    df_test = pd.DataFrame(test_data).set_index("image_id")

    # ToDo
    # df_counts = df_train.groupby("category_id").size()
    # labels = list(df_counts.index)
    # sampler = WeightedRandomSampler(torch.from_numpy(1. / df_counts.values), len(df_counts))

    df_train, df_val = train_test_split(df_train, test_size=val_split, stratify=df_train["category_id"].tolist())
    # noisy predictions shall not be in validation
    df_train = append_predictions(df_train, path_csv=predict_csv)

    datamodule = ImageClassificationData.from_data_frame(
        input_field="file_name",
        target_fields="category_id",
        # for simplicity take just half of the data
        # train_data_frame=df_train[:len(df_train) // 2],
        train_data_frame=df_train,
        train_images_root=dataset_dir,
        val_data_frame=df_val,
        val_images_root=dataset_dir,
        transform=ImageClassificationInputTransform,
        transform_kwargs={"image_size": (image_size, image_size)},
        batch_size=batch_size,
        num_workers=num_workers if num_workers else min(batch_size, int(os.cpu_count() / gpus)),
        # sampler=sampler,
    )

    loss = LabelSmoothingCrossEntropy(label_smoothing) if label_smoothing else AsymmetricLossSingleLabel()

    model = ImageClassifier(
        backbone=model_backbone,
        metrics=F1Score(num_classes=datamodule.num_classes, average="macro"),
        pretrained=model_pretrained,
        loss_fn=loss,
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

    trainer_flags = dict(
        callbacks=cbs,
        max_epochs=max_epochs,
        precision="bf16" if gpus else 32,
        gpus=gpus,
        accelerator="ddp" if gpus > 1 else None,
        logger=logger,
        gradient_clip_val=1e-2,
    )
    trainer_flags.update(trainer_kwargs)
    trainer = flash.Trainer(**trainer_flags)

    # Train the model
    # trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
    trainer.finetune(model, datamodule=datamodule, strategy=("freeze_unfreeze", 2))

    # Save the model!
    checkpoint_name = f"herbarium-classif-{log_id}_{model_backbone}-{image_size}px.pt"
    trainer.save_checkpoint(os.path.join(checkpoints_dir, checkpoint_name))

    if run_inference and trainer.is_global_zero:
        df_preds = inference(
            model, df_test, dataset_dir=dataset_dir, image_size=image_size, batch_size=batch_size, gpus=gpus
        )
        preds_name = f"predictions_herbarium-{log_id}_{model_backbone}-{image_size}.csv"
        df_preds.to_csv(os.path.join(checkpoints_dir, preds_name))
        submission = pd.DataFrame({"Id": df_preds.index, "Predicted": df_preds["category_id"]}).set_index("Id")
        submission_name = f"submission_herbarium-{log_id}_{model_backbone}-{image_size}.csv"
        submission.to_csv(os.path.join(checkpoints_dir, submission_name))


if __name__ == "__main__":
    fire.Fire(main)

import itertools
import multiprocessing as mproc
import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

TRAIN_TRANSFORM = T.Compose([
    T.Resize(512),
    T.RandomPerspective(),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # T.Normalize([0.431, 0.498,  0.313], [0.237, 0.239, 0.227]),  # custom
])

VALID_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # T.Normalize([0.431, 0.498,  0.313], [0.237, 0.239, 0.227]),  # custom
])


class PlantPathologyDataset(Dataset):
    """The ful dataset with one-hot encoding for multi-label case."""

    def __init__(
        self,
        path_csv: str = 'train.csv',
        path_img_dir: str = 'train_images',
        transforms=None,
        mode: str = 'train',
        split: float = 0.8,
        uq_labels: Tuple[str] = None,
    ):
        self.path_img_dir = path_img_dir
        self.transforms = transforms
        self.mode = mode

        self.data = pd.read_csv(path_csv)
        if uq_labels:
            self.labels_unique = uq_labels
        else:
            labels_all = list(itertools.chain(*[lbs.split(" ") for lbs in self.data['labels']]))
            self.labels_unique = sorted(set(labels_all))
        self.labels_lut = {lb: i for i, lb in enumerate(self.labels_unique)}
        self.num_classes = len(self.labels_unique)
        # shuffle data
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

        # split dataset
        assert 0.0 <= split <= 1.0
        frac = int(split * len(self.data))
        self.data = self.data[:frac] if mode == 'train' else self.data[frac:]
        self.img_names = list(self.data['image'])
        self.labels = list(self.data['labels'])

    def to_one_hot(self, labels: str) -> tuple:
        one_hot = [0] * len(self.labels_unique)
        for lb in labels.split(" "):
            one_hot[self.labels_lut[lb]] = 1
        return tuple(one_hot)

    def __getitem__(self, idx: int) -> tuple:
        img_path = os.path.join(self.path_img_dir, self.img_names[idx])
        assert os.path.isfile(img_path)
        label = self.labels[idx]
        img = plt.imread(img_path)

        # augmentation
        if self.transforms:
            img = self.transforms(Image.fromarray(img))
        label = self.to_one_hot(label)
        return img, torch.tensor(label)

    def __len__(self) -> int:
        return len(self.data)


class PlantPathologySimpleDataset(PlantPathologyDataset):
    """Simplified version; we keep only complex label for multi-label cases and the true label for all others."""

    def __getitem__(self, idx: int) -> tuple:
        img, label = super().__getitem__(idx)
        if torch.sum(label) > 1:
            label = self.labels_lut['complex']
        else:
            label = torch.argmax(label)
        return img, int(label)


class PlantPathologyDM(LightningDataModule):

    def __init__(
        self,
        path_csv: str = 'train.csv',
        path_img_dir: str = 'train_images',
        batch_size: int = 128,
        num_workers: int = None,
        simple: bool = False,
    ):
        super().__init__()
        self.path_csv = path_csv
        self.path_img_dir = path_img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else mproc.cpu_count()
        self.train_dataset = None
        self.valid_dataset = None
        self.dataset_cls = PlantPathologySimpleDataset if simple else PlantPathologyDataset

    def prepare_data(self):
        pass

    @property
    def num_classes(self) -> int:
        assert self.train_dataset and self.valid_dataset
        return max(self.train_dataset.num_classes, self.valid_dataset.num_classes)

    def setup(self, stage=None):
        ds = self.dataset_cls(self.path_csv, self.path_img_dir, mode='train', split=1.0)
        self.train_dataset = self.dataset_cls(
            self.path_csv, self.path_img_dir, mode='train', uq_labels=ds.labels_unique, transforms=TRAIN_TRANSFORM
        )
        print(f"training dataset: {len(self.train_dataset)}")
        self.valid_dataset = self.dataset_cls(
            self.path_csv, self.path_img_dir, mode='valid', uq_labels=ds.labels_unique, transforms=VALID_TRANSFORM
        )
        print(f"validation dataset: {len(self.valid_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        pass

import logging
import multiprocessing as mproc
import os
from math import ceil

import matplotlib.pylab as plt
import pandas as pd
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
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.Normalize([0.431, 0.498, 0.313], [0.237, 0.239, 0.227]),
])

VALID_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.Normalize([0.431, 0.498, 0.313], [0.237, 0.239, 0.227]),
])


class CassavaDataset(Dataset):

    def __init__(
        self,
        path_csv: str = "/content/train.csv",
        path_img_dir: str = "/content/train_images/",
        transforms=None,
        mode: str = 'train',
        split: float = 0.8,
    ):
        self.path_img_dir = path_img_dir
        self.transforms = transforms
        self.mode = mode

        self.data = pd.read_csv(path_csv)
        # shuffle data
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

        # split dataset
        assert 0.0 <= split <= 1.0
        frac = int(ceil(split * len(self.data)))
        self.data = self.data[:frac] if mode == 'train' else self.data[frac:]
        self.img_names = list(self.data['image_id'])
        self.labels = list(self.data['label'])

    def __getitem__(self, idx: int) -> tuple:
        img_path = os.path.join(self.path_img_dir, self.img_names[idx])
        assert os.path.isfile(img_path)
        label = self.labels[idx]
        img = plt.imread(img_path)

        # augmentation
        if self.transforms:
            img = self.transforms(Image.fromarray(img))
        return img, label

    def __len__(self) -> int:
        return len(self.data)


class CassavaDataModule(LightningDataModule):

    def __init__(
        self,
        path_csv: str = "/content/train.csv",
        path_img_dir: str = "/content/train_images/",
        train_augment=TRAIN_TRANSFORM,
        valid_augment=VALID_TRANSFORM,
    ):
        super().__init__()
        self.path_csv = path_csv
        self.path_img_dir = path_img_dir
        self.train_augment = train_augment
        self.valid_augment = valid_augment

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = CassavaDataset(
            self.path_csv,
            self.path_img_dir,
            mode='train',
            transforms=self.train_augment,
        )
        logging.info(f"training dataset: {len(self.train_dataset)}")
        self.valid_dataset = CassavaDataset(
            self.path_csv,
            self.path_img_dir,
            mode='valid',
            transforms=self.valid_augment,
        )
        logging.info(f"validation dataset: {len(self.valid_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=128,
            num_workers=mproc.cpu_count(),
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=128,
            num_workers=mproc.cpu_count(),
            shuffle=False,
        )

    def test_dataloader(self):
        pass

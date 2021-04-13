import glob
import itertools
import logging
import multiprocessing as mproc
import os
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

#: feasible image extension for testing
from kaggle_plantpatho.augment import KORNIA_TRAIN_TRANSFORM, KORNIA_VALID_TRANSFORM

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


class PlantPathologyDataset(Dataset):
    """The ful dataset with one-hot encoding for multi-label case."""

    def __init__(
        self,
        df_data: Union[str, pd.DataFrame] = 'train.csv',
        path_img_dir: str = 'train_images',
        transforms=None,
        mode: str = 'train',
        split: float = 0.8,
        uq_labels: Tuple[str] = None,
        random_state=42,
    ):
        self.path_img_dir = path_img_dir
        self.transforms = transforms
        self.mode = mode

        # set or load the config table
        if isinstance(df_data, pd.DataFrame):
            self.data = df_data
        elif isinstance(df_data, str):
            assert os.path.isfile(df_data), f"missing file: {df_data}"
            self.data = pd.read_csv(df_data)
        else:
            raise ValueError(f'unrecognised input for DataFrame/CSV: {df_data}')

        # take over existing table or load from file
        if uq_labels:
            self.labels_unique = uq_labels
        else:
            labels_all = list(itertools.chain(*[lbs.split(" ") for lbs in self.data['labels']]))
            self.labels_unique = sorted(set(labels_all))
        self.labels_lut = {lb: i for i, lb in enumerate(self.labels_unique)}
        self.num_classes = len(self.labels_unique)

        # shuffle data
        self.data = self.data.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # split dataset
        assert 0.0 <= split <= 1.0, f"split {split} is out of range"
        frac = int(split * len(self.data))
        self.data = self.data[:frac] if mode == 'train' else self.data[frac:]
        self.img_names = list(self.data['image'])
        self.labels = list(self.data['labels'])

    def to_onehot_encoding(self, labels: str) -> tuple:
        # processed with encoding
        one_hot = [0] * len(self.labels_unique)
        for lb in labels.split(" "):
            one_hot[self.labels_lut[lb]] = 1
        return tuple(one_hot)

    def __getitem__(self, idx: int) -> tuple:
        img_name = self.img_names[idx]
        img_path = os.path.join(self.path_img_dir, img_name)
        assert os.path.isfile(img_path)
        label = self.labels[idx]
        img = plt.imread(img_path)

        # augmentation
        if self.transforms:
            img = self.transforms(Image.fromarray(img))
        # in case of predictions, return image name as label
        label = torch.tensor(self.to_onehot_encoding(label)) if label else img_name
        return img, label

    def __len__(self) -> int:
        return len(self.data)


class PlantPathologySimpleDataset(PlantPathologyDataset):
    """Simplified version; we keep only complex label for multi-label cases and the true label for all others."""

    def __getitem__(self, idx: int) -> tuple:
        img, label = super().__getitem__(idx)
        # shortcut for prediction without labels
        if isinstance(label, str):
            return img, label
        # get complex or find the one...
        if torch.sum(label) > 1:
            label = self.labels_lut['complex']
        else:
            label = torch.argmax(label)
        return img, int(label)


class PlantPathologyDM(LightningDataModule):

    def __init__(
        self,
        path_csv: str = 'train.csv',
        base_path: str = '.',
        batch_size: int = 128,
        num_workers: int = None,
        simple: bool = False,
        train_transforms=None,
        valid_transforms=None,
        split: float = 0.8,
    ):
        super().__init__()
        # path configurations
        assert os.path.isdir(base_path), f"missing folder: {base_path}"
        self.train_dir = os.path.join(base_path, 'train_images')
        self.test_dir = os.path.join(base_path, 'test_images')

        if not os.path.isfile(path_csv):
            path_csv = os.path.join(base_path, path_csv)
        assert os.path.isfile(path_csv), f"missing table: {path_csv}"
        self.path_csv = path_csv

        # other configs
        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers if num_workers is not None else mproc.cpu_count()
        self.labels_unique: Sequence = ...
        self.lut_label: Dict = ...

        # need to be filled in setup()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_table = []
        self.test_dataset = None
        self.train_transforms = train_transforms or KORNIA_TRAIN_TRANSFORM
        self.valid_transforms = valid_transforms or KORNIA_VALID_TRANSFORM
        self.dataset_cls: Type = PlantPathologySimpleDataset if simple else PlantPathologyDataset

    def prepare_data(self):
        pass

    @property
    def num_classes(self) -> int:
        assert self.train_dataset and self.valid_dataset
        return max(self.train_dataset.num_classes, self.valid_dataset.num_classes)

    def onehot_to_labels(self, onehot: Tensor, thr: float = 0.5, label_required: bool = True) -> Union[str, List[str]]:
        """Convert Model outputs to string labels

        Args:
            onehot: one-hot encoding
            thr: threshold for label binarization
            label_required: if it is required to return any label and no label is above `thr`, use argmax
        """
        assert self.lut_label
        # on case it is not one hot encoding but single label
        if onehot.nelement() == 1:
            return self.lut_label[onehot[0]]
        labels = [self.lut_label[i] for i, s in enumerate(onehot) if s >= thr]
        # in case no reached threshold then take max
        if not labels and label_required:
            idx = torch.argmax(onehot).item()
            labels = [self.lut_label[idx]]
        return sorted(labels)

    def setup(self, *_, **__) -> None:
        """Prepare datasets"""
        assert os.path.isdir(self.train_dir), f"missing folder: {self.train_dir}"
        ds = self.dataset_cls(self.path_csv, self.train_dir, mode='train', split=1.0)
        self.labels_unique = ds.labels_unique
        self.lut_label = dict(enumerate(self.labels_unique))

        ds_defaults = dict(
            df_data=self.path_csv,
            path_img_dir=self.train_dir,
            split=self.split,
            uq_labels=self.labels_unique,
        )
        self.train_dataset = self.dataset_cls(**ds_defaults, mode='train', transforms=self.train_transforms)
        logging.info(f"training dataset: {len(self.train_dataset)}")
        self.valid_dataset = self.dataset_cls(**ds_defaults, mode='valid', transforms=self.valid_transforms)
        logging.info(f"validation dataset: {len(self.valid_dataset)}")

        if not os.path.isdir(self.test_dir):
            return
        ls_images = glob.glob(os.path.join(self.test_dir, '*.*'))
        ls_images = [os.path.basename(p) for p in ls_images if os.path.splitext(p)[-1] in IMAGE_EXTENSIONS]
        self.test_table = [dict(image=n, labels='') for n in ls_images]
        self.test_dataset = self.dataset_cls(
            df_data=pd.DataFrame(self.test_table),
            path_img_dir=self.test_dir,
            split=0,
            uq_labels=self.labels_unique,
            mode='test',
            transforms=self.valid_transforms
        )
        logging.info(f"test dataset: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=False,
            )
        logging.warning('no testing images found')

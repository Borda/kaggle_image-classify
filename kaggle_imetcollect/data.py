import glob
import itertools
import logging
import multiprocessing as mproc
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from joblib import delayed, Parallel
from PIL import Image, ImageFile
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

#: default training augmentation
TORCHVISION_TRAIN_TRANSFORM = T.Compose(
    [
        T.Resize(size=512, interpolation=Image.BILINEAR),
        T.RandomRotation(degrees=30),
        T.RandomPerspective(distortion_scale=0.2),
        T.RandomResizedCrop(size=224),
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(p=0.5),
        # T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
#: default validation augmentation
TORCHVISION_VALID_TRANSFORM = T.Compose(
    [
        T.Resize(size=256, interpolation=Image.BILINEAR),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_nb_pixels(img_path: str):
    try:
        img = Image.open(img_path)
        return np.prod(img.size)
    except Exception:
        return 0


class IMetDataset(Dataset):
    """The ful dataset with one-hot encoding for multi-label case."""

    IMAGE_SIZE_LIMIT = 1000
    COL_LABELS = "attribute_ids"
    COL_IMAGES = "id"

    def __init__(
        self,
        df_data: Union[str, pd.DataFrame] = "train-from-kaggle.csv",
        path_img_dir: str = "train-1/train-1",
        transforms=None,
        mode: str = "train",
        split: float = 0.8,
        uq_labels: Tuple[str] = None,
        random_state: Optional[int] = None,
        check_imgs: bool = True,
    ):
        self.path_img_dir = path_img_dir
        self.transforms = transforms
        self.mode = mode
        self._img_names = None
        self._raw_labels = None

        # set or load the config table
        if isinstance(df_data, pd.DataFrame):
            self.data = df_data
        elif isinstance(df_data, str):
            assert os.path.isfile(df_data), f"missing file: {df_data}"
            self.data = pd.read_csv(df_data)
        else:
            raise ValueError(f"unrecognised input for DataFrame/CSV: {df_data}")

        # take over existing table or load from file
        if uq_labels:
            self.labels_unique = tuple(uq_labels)
        else:
            labels_all = list(itertools.chain(*[lbs.split(" ") for lbs in self.raw_labels]))
            # labels_all = [int(lb) for lb in labels_all]
            self.labels_unique = tuple(sorted(set(labels_all)))
        self.labels_lut = {lb: i for i, lb in enumerate(self.labels_unique)}
        self.num_classes = len(self.labels_unique)

        # filter/drop too small images
        if check_imgs:
            with Parallel(n_jobs=mproc.cpu_count()) as parallel:
                self.data["pixels"] = parallel(
                    delayed(get_nb_pixels)(os.path.join(self.path_img_dir, im)) for im in self.img_names
                )
            nb_small_imgs = sum(self.data["pixels"] < self.IMAGE_SIZE_LIMIT)
            if nb_small_imgs:
                logging.warning(f"found and dropped {nb_small_imgs} too small or invalid images :/")
            self.data = self.data[self.data["pixels"] >= self.IMAGE_SIZE_LIMIT]
        # shuffle data
        if random_state is not None:
            self.data = self.data.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # split dataset
        assert 0.0 <= split <= 1.0, f"split {split} is out of range"
        frac = int(split * len(self.data))
        self.data = self.data[:frac] if mode == "train" else self.data[frac:]
        # need to reset after another split since it cached
        self._img_names = None
        self._raw_labels = None
        self.labels = self._prepare_labels()

    @property
    def img_names(self):
        if not self._img_names:
            self._img_names = [f"{n}.png" if "." not in n else n for n in self.data[self.COL_IMAGES]]
        return self._img_names

    @property
    def raw_labels(self):
        if not self._raw_labels:
            self._raw_labels = list(self.data[self.COL_LABELS])
        return self._raw_labels

    def _prepare_labels(self) -> list:
        return [torch.tensor(self.to_onehot_encoding(lb)) if lb else None for lb in self.raw_labels]

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
        # todo: find some faster way, do conversion only if needed; im.mode not in ("L", "RGB")
        img = Image.open(img_path).convert("RGB")

        # augmentation
        if self.transforms:
            img = self.transforms(img)

        # in case of predictions, return image name as label
        label = label if label is not None else img_name
        return img, label

    def __len__(self) -> int:
        return len(self.data)


class IMetDM(LightningDataModule):
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        base_path: str,
        path_csv: str = "train-from-kaggle.csv",
        batch_size: int = 128,
        num_workers: int = None,
        train_transforms=TORCHVISION_TRAIN_TRANSFORM,
        valid_transforms=TORCHVISION_VALID_TRANSFORM,
        split: float = 0.8,
    ):
        super().__init__()
        # path configurations
        assert os.path.isdir(base_path), f"missing folder: {base_path}"
        self.train_dir = os.path.join(base_path, "train-1/train-1")
        self.test_dir = os.path.join(base_path, "test/test")

        if not os.path.isfile(path_csv):
            path_csv = os.path.join(base_path, path_csv)
        assert os.path.isfile(path_csv), f"missing table: {path_csv}"
        self.path_csv = path_csv

        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

        # other configs
        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers if num_workers is not None else mproc.cpu_count()
        self.labels_unique: Sequence = ...
        self.lut_label: Dict = ...
        self.label_histogram: Tensor = ...

        # need to be filled in setup()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_table = []
        self.test_dataset = None

    def prepare_data(self):
        pass

    @property
    def num_classes(self) -> int:
        return len(self.labels_unique)

    @staticmethod
    def onehot_mapping(
        onehot: Tensor,
        lut_label: Dict[int, str],
        thr: float = 0.5,
        label_required: bool = True,
    ) -> Union[str, List[str]]:
        """Convert Model outputs to string labels.

        Args:
            onehot: one-hot encoding
            lut_label: look-up-table with labels
            thr: threshold for label binarization
            label_required: if it is required to return any label and no label is above `thr`, use argmax
        """
        assert lut_label
        # on case it is not one hot encoding but single label
        if onehot.nelement() == 1:
            return lut_label[onehot[0]]
        labels = [lut_label[i] for i, s in enumerate(onehot) if s >= thr]
        # in case no reached threshold then take max
        if not labels and label_required:
            idx = torch.argmax(onehot).item()
            labels = [lut_label[idx]]
        return sorted(labels)

    def onehot_to_labels(
        self, onehot: Tensor, thr: float = 0.5, with_sigm: bool = True, label_required: bool = True
    ) -> Union[str, List[str]]:
        """Convert Model outputs to string labels.

        Args:
            onehot: one-hot encoding
            thr: threshold for label binarization
            with_sigm: apply sigmoid to convert to probabilities
            label_required: if it is required to return any label and no label is above `thr`, use argmax
        """
        if with_sigm:
            onehot = torch.sigmoid(onehot)
        return self.onehot_mapping(onehot, self.lut_label, thr=thr, label_required=label_required)

    def setup(self, *_, **__) -> None:
        """Prepare datasets."""
        pbar = tqdm.tqdm(total=4)
        assert os.path.isdir(self.train_dir), f"missing folder: {self.train_dir}"
        ds = IMetDataset(self.path_csv, self.train_dir, mode="train", split=1.0)
        self.labels_unique = ds.labels_unique
        self.lut_label = dict(enumerate(self.labels_unique))
        pbar.update()

        ds_defaults = dict(
            df_data=ds.data,
            path_img_dir=self.train_dir,
            split=self.split,
            uq_labels=self.labels_unique,
            check_imgs=False,
        )
        self.train_dataset = IMetDataset(**ds_defaults, mode="train", transforms=self.train_transforms)
        logging.info(f"training dataset: {len(self.train_dataset)}")
        pbar.update()
        self.valid_dataset = IMetDataset(**ds_defaults, mode="valid", transforms=self.valid_transforms)
        logging.info(f"validation dataset: {len(self.valid_dataset)}")
        pbar.update()

        if not os.path.isdir(self.test_dir):
            return
        ls_images = glob.glob(os.path.join(self.test_dir, "*.*"))
        ls_images = [os.path.basename(p) for p in ls_images if os.path.splitext(p)[-1] in self.IMAGE_EXTENSIONS]
        self.test_table = [{"id": n, "attribute_ids": ""} for n in ls_images]
        self.test_dataset = IMetDataset(
            df_data=pd.DataFrame(self.test_table),
            path_img_dir=self.test_dir,
            split=0,
            uq_labels=self.labels_unique,
            mode="test",
            transforms=self.valid_transforms,
        )
        logging.info(f"test dataset: {len(self.test_dataset)}")
        pbar.update()

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
                num_workers=self.num_workers,
                shuffle=False,
            )
        logging.warning("no testing images found")

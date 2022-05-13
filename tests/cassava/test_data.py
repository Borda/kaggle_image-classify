import os

import numpy

from kaggle_imgclassif.cassava.data import CassavaDataModule, CassavaDataset

from tests import _ROOT_DATA

PATH_DATA = os.path.join(_ROOT_DATA, "cassava")


def test_dataset(path_data=PATH_DATA):
    dataset = CassavaDataset(
        path_csv=os.path.join(path_data, "train.csv"),
        path_img_dir=os.path.join(path_data, "train_images"),
    )
    img, lb = dataset[0]
    assert isinstance(img, numpy.ndarray)


def test_datamodule(path_data=PATH_DATA):
    dm = CassavaDataModule(
        path_csv=os.path.join(path_data, "train.csv"),
        path_img_dir=os.path.join(path_data, "train_images"),
    )
    dm.setup()

    for imgs, lbs in dm.train_dataloader():
        assert len(imgs)
        assert len(lbs)
        break

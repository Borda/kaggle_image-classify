import os

import numpy

from kaggle_cassava.data import CassavaDataModule, CassavaDataset

_PATH_HERE = os.path.dirname(__file__)


def test_dataset(root_path=_PATH_HERE):
    dataset = CassavaDataset(
        path_csv=os.path.join(root_path, "data", "train.csv"),
        path_img_dir=os.path.join(root_path, "data", "train_images")
    )
    img, lb = dataset[0]
    assert isinstance(img, numpy.ndarray)


def test_datamodule(root_path=_PATH_HERE):
    dm = CassavaDataModule(
        path_csv=os.path.join(root_path, "data", "train.csv"),
        path_img_dir=os.path.join(root_path, "data", "train_images")
    )
    dm.setup()

    for imgs, lbs in dm.train_dataloader():
        assert len(imgs)
        assert len(lbs)
        break

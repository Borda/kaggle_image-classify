import os

import numpy

from kaggle_cassava.data import CassavaDataModule, CassavaDataset

from tests import _ROOT_TESTS


def test_dataset(root_path=_ROOT_TESTS):
    dataset = CassavaDataset(
        path_csv=os.path.join(root_path, "data_cassava", "train.csv"),
        path_img_dir=os.path.join(root_path, "data_cassava", "train_images"),
    )
    img, lb = dataset[0]
    assert isinstance(img, numpy.ndarray)


def test_datamodule(root_path=_ROOT_TESTS):
    dm = CassavaDataModule(
        path_csv=os.path.join(root_path, "data_cassava", "train.csv"),
        path_img_dir=os.path.join(root_path, "data_cassava", "train_images"),
    )
    dm.setup()

    for imgs, lbs in dm.train_dataloader():
        assert len(imgs)
        assert len(lbs)
        break

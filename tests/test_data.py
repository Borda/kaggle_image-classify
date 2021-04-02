import os

import numpy

from kaggle_plantpatho.data import PlantPathologyDataset, PlantPathologyDM, PlantPathologySimpleDataset

_PATH_HERE = os.path.dirname(__file__)


def test_dataset(root_path=_PATH_HERE):
    dataset = PlantPathologyDataset(
        path_csv=os.path.join(root_path, "data", "train.csv"),
        path_img_dir=os.path.join(root_path, "data", "train_images")
    )
    img, lb = dataset[0]
    assert isinstance(img, numpy.ndarray)


def test_dataset_simple(root_path=_PATH_HERE):
    dataset = PlantPathologySimpleDataset(
        path_csv=os.path.join(root_path, "data", "train.csv"),
        path_img_dir=os.path.join(root_path, "data", "train_images")
    )
    img, lb = dataset[0]
    assert isinstance(img, numpy.ndarray)


def test_datamodule(root_path=_PATH_HERE):
    dm = PlantPathologyDM(
        path_csv=os.path.join(root_path, "data", "train.csv"),
        path_img_dir=os.path.join(root_path, "data", "train_images")
    )
    dm.setup()

    for imgs, lbs in dm.train_dataloader():
        assert len(imgs)
        assert len(lbs)
        break

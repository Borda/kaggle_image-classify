import os

import numpy
import pytest

from kaggle_plantpatho.data import PlantPathologyDataset, PlantPathologyDM, PlantPathologySimpleDataset

_PATH_HERE = os.path.dirname(__file__)


@pytest.mark.parametrize("data_cls", [PlantPathologyDataset, PlantPathologySimpleDataset])
def test_dataset(data_cls, root_path=_PATH_HERE):
    dataset = PlantPathologyDataset(
        path_csv=os.path.join(root_path, "data", "train.csv"),
        path_img_dir=os.path.join(root_path, "data", "train_images")
    )
    img, lb = dataset[0]
    assert isinstance(img, numpy.ndarray)


@pytest.mark.parametrize("simple", [True, False])
def test_datamodule(simple, root_path=_PATH_HERE):
    dm = PlantPathologyDM(
        path_csv=os.path.join(root_path, "data", "train.csv"),
        path_img_dir=os.path.join(root_path, "data", "train_images"),
        simple=simple,
    )
    dm.setup()

    for imgs, lbs in dm.train_dataloader():
        assert len(imgs)
        assert len(lbs)
        break

import os

import numpy
import pytest
from torch import Tensor

from kaggle_plantpatho.data import PlantPathologyDataset, PlantPathologyDM, PlantPathologySimpleDataset

_PATH_HERE = os.path.dirname(__file__)


@pytest.mark.parametrize("data_cls", [PlantPathologyDataset, PlantPathologySimpleDataset])
def test_dataset(data_cls, root_path=_PATH_HERE):
    dataset = PlantPathologyDataset(
        df_data=os.path.join(root_path, "data", "train.csv"),
        path_img_dir=os.path.join(root_path, "data", "train_images")
    )
    img, lb = dataset[0]
    assert isinstance(img, numpy.ndarray)


@pytest.mark.parametrize("simple", [True, False])
def test_datamodule(simple, root_path=_PATH_HERE):
    dm = PlantPathologyDM(
        path_csv=os.path.join(root_path, "data", "train.csv"),
        base_path=os.path.join(root_path, "data"),
        simple=simple,
        split=0.6,
    )
    dm.setup()
    assert dm.labels_unique
    assert dm.lut_label
    assert isinstance(dm.label_histogram, Tensor)

    for imgs, lbs in dm.train_dataloader():
        assert len(imgs)
        assert len(lbs)
        break

    for imgs, lbs in dm.val_dataloader():
        assert len(imgs)
        assert len(lbs)
        break

    for imgs, names in dm.test_dataloader():
        assert len(imgs)
        assert len(names)
        assert isinstance(names[0], str)
        break

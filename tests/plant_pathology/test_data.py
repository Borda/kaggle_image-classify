import os

import numpy
import pytest

from kaggle_plantpathology.data import PlantPathologyDataset, PlantPathologyDM, PlantPathologySimpleDataset
from torch import Tensor

_PATH_HERE = os.path.dirname(__file__)
_TEST_IMAGE_NAMES = (
    "800113bb65efe69e.jpg",
    "8002cb321f8bfcdf.jpg",
    "8a2d598f2ec436e6.jpg",
    "800f85dc5f407aef.jpg",
    "8a1a97abda0b4a7a.jpg",
    "8a0be55d81f4bf0c.jpg",
    "8a954b82bf81f2bc.jpg",
)
_TEST_UNIQUE_LABELS = (
    "cider_apple_rust",
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
)
_TEST_LABELS_BINARY = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
]


@pytest.mark.parametrize(
    "data_cls,labels",
    [
        (PlantPathologyDataset, _TEST_LABELS_BINARY),
        (PlantPathologySimpleDataset, [3, 1, 4, 0, 2, 3, 1]),
    ],
)
@pytest.mark.parametrize("phase", ["train", "valid"])
def test_dataset(data_cls, labels, phase, root_path=_PATH_HERE):
    dataset = data_cls(
        df_data=os.path.join(root_path, "data_plant-pathology", "train.csv"),
        path_img_dir=os.path.join(root_path, "data_plant-pathology", "train_images"),
        split=1.0 if phase == "train" else 0.0,
        mode=phase,
    )
    assert len(dataset) == 7
    img, _ = dataset[0]
    assert isinstance(img, numpy.ndarray)
    assert _TEST_IMAGE_NAMES == tuple(dataset.img_names) == tuple(dataset.data["image"])
    assert _TEST_UNIQUE_LABELS == dataset.labels_unique
    lbs = [dataset[i][1] for i in range(len(dataset))]
    if isinstance(lbs[0], Tensor):
        lbs = [list(lb.numpy()) for lb in lbs]
    # mm = lambda lb: np.array([i for i, l in enumerate(lb) if l])
    # lb_names = [np.array(dataset.labels_unique)[mm(lb)] for lb in lbs]
    assert labels == lbs


@pytest.mark.parametrize("simple", [True, False])
@pytest.mark.parametrize("balance", [True, False])
def test_datamodule(simple, balance, root_path=_PATH_HERE):
    dm = PlantPathologyDM(
        path_csv="train.csv",
        base_path=os.path.join(root_path, "data_plant-pathology"),
        simple=simple,
        split=0.6,
        balancing=balance,
    )
    dm.setup()
    assert dm.num_classes > 0
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

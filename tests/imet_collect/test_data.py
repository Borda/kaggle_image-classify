import os

import pytest
import torch
from PIL import Image
from torch import tensor

from kaggle_imgclassif.imet_collect.data import IMetDataset, IMetDM
from tests import _ROOT_DATA

PATH_DATA = os.path.join(_ROOT_DATA, "imet-collect")
_TEST_IMAGE_NAMES = (
    "1cc66a822733a3c3a1ce66fe4be60a6f",
    "09fe6ff247881b37779bcb386c26d7bb",
    "258e4a904729119efd85faaba80c965a",
    "11a87738861970a67249592db12f2da1",
    "12c80004e34f9102cad72c7312133529",
    "0d5b8274de10cd73836c858c101266ea",
    "14f3fa3b620d46be00696eacda9df583",
)
_TEST_UNIQUE_LABELS = (
    "124",
    "1660",
    "2281",
    "233",
    "2362",
    "262",
    "2941",
    "3192",
    "3193",
    "3235",
    "3334",
    "341",
    "3465",
    "370",
    "507",
    "782",
    "783",
    "784",
    "792",
    "946",
    "96",
)
_TEST_LABELS_BINARY = [
    tensor([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    tensor([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
    tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    tensor([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
    tensor([0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]),
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
]


@pytest.mark.parametrize("phase", ["train", "valid"])
def test_dataset(phase, path_data=PATH_DATA):
    dataset = IMetDataset(
        df_data=os.path.join(path_data, "train-from-kaggle.csv"),
        path_img_dir=os.path.join(path_data, "train-1", "train-1"),
        split=1.0 if phase == "train" else 0.0,
        mode=phase,
        random_state=42,
    )
    assert len(dataset) == 7
    img, _ = dataset[0]
    assert isinstance(img, Image.Image)
    _img_names = [os.path.splitext(im)[0] for im in dataset.img_names]
    assert tuple(_img_names) == tuple(dataset.data["id"]) == _TEST_IMAGE_NAMES
    assert dataset.labels_unique == _TEST_UNIQUE_LABELS
    lbs = [tensor(dataset[i][1]) for i in range(len(dataset))]
    # mm = lambda lb: np.array([i for i, l in enumerate(lb) if l])
    # lb_names = [np.array(dataset.labels_unique)[mm(lb)] for lb in lbs]
    assert all(torch.equal(a, b) for a, b in zip(_TEST_LABELS_BINARY, lbs))


def test_datamodule(path_data=PATH_DATA):
    dm = IMetDM(
        path_csv="train-from-kaggle.csv",
        base_path=path_data,
        batch_size=2,
        split=0.6,
    )
    dm.setup()
    assert dm.num_classes == len(_TEST_UNIQUE_LABELS)
    assert dm.labels_unique == _TEST_UNIQUE_LABELS
    assert len(dm.lut_label) == len(_TEST_UNIQUE_LABELS)
    # assert isinstance(dm.label_histogram, Tensor)

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

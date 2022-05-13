import os

import timm
from pytorch_lightning import Trainer

from kaggle_imgclassif.imet_collect.data import IMetDM
from kaggle_imgclassif.imet_collect.models import LitMet

from tests import _ROOT_DATA

PATH_DATA = os.path.join(_ROOT_DATA, "imet-collect")


def test_create_model():
    net = timm.create_model("resnet34", pretrained=False, num_classes=5)
    LitMet(model=net, num_classes=5)


def test_devel_run(tmpdir, path_data=PATH_DATA):
    """Sample fast dev run..."""
    dm = IMetDM(
        path_csv=os.path.join(path_data, "train-from-kaggle.csv"),
        base_path=path_data,
        batch_size=2,
        split=0.6,
    )
    dm.setup()
    net = timm.create_model("resnet18", num_classes=dm.num_classes)
    model = LitMet(model=net, num_classes=dm.num_classes)

    # smoke run
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )
    trainer.fit(model, datamodule=dm)

    # test predictions
    for imgs, names in dm.test_dataloader():
        encode = model(imgs)
        # it has only batch size 1
        for oh, name in zip(encode, names):
            dm.binary_encoding_to_labels(oh)

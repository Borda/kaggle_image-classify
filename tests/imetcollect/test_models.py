import os

import pytest

from kaggle_imetcollect.data import IMetDM
from kaggle_imetcollect.models import LitMet, LitResnet
from pytorch_lightning import Trainer

from tests import _ROOT_TESTS


def test_create_resnet():
    LitResnet(arch="resnet18")


@pytest.mark.parametrize("model_cls", [LitResnet])
def test_create_model(model_cls):
    net = LitResnet(arch="resnet18")
    LitMet(model=net)


def test_devel_run(tmpdir, root_path=_ROOT_TESTS):
    """Sample fast dev run..."""
    dm = IMetDM(
        path_csv=os.path.join(root_path, "data_imet-collect", "train-from-kaggle.csv"),
        base_path=os.path.join(root_path, "data_imet-collect"),
        batch_size=2,
        split=0.6,
    )
    dm.setup()
    net = LitResnet(arch="resnet18", num_classes=dm.num_classes)
    model = LitMet(model=net)

    # smoke run
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )
    trainer.fit(model, datamodule=dm)

    # test predictions
    for imgs, names in dm.test_dataloader():
        onehots = model(imgs)
        # it has only batch size 1
        for oh, name in zip(onehots, names):
            dm.onehot_to_labels(oh)

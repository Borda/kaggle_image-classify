import os

import pytest
from pytorch_lightning import Trainer

from kaggle_plantpatho.data import PlantPathologyDM
from kaggle_plantpatho.models import LitPlantPathology, LitResnet, MultiPlantPathology

_PATH_HERE = os.path.dirname(__file__)


def test_create_resnet():
    LitResnet(arch='resnet18')


@pytest.mark.parametrize("model_cls", [LitPlantPathology, MultiPlantPathology])
def test_create_model(model_cls):
    net = LitResnet(arch='resnet18')
    LitPlantPathology(model=net)


@pytest.mark.parametrize("ds_simple,model_cls", [
    (True, LitPlantPathology),
    (False, MultiPlantPathology),
])
def test_devel_run(tmpdir, ds_simple, model_cls, root_path=_PATH_HERE):
    """Sample fast dev run..."""
    dm = PlantPathologyDM(
        path_csv=os.path.join(root_path, "data", "train.csv"),
        base_path=os.path.join(root_path, "data"),
        simple=ds_simple,
        batch_size=2,
        split=0.6,
    )
    dm.setup()
    net = LitResnet(arch='resnet18', num_classes=dm.num_classes)
    model = model_cls(model=net)

    # smoke run
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )
    trainer.fit(model, datamodule=dm)

    # test predictions
    for imgs, _ in dm.test_dataloader():
        onehot = model(imgs)
        dm.onehot_to_labels(onehot)

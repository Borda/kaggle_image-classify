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


@pytest.mark.parametrize("ds_simple,model_cls", [(True, LitPlantPathology)])  # , (False, MultiPlantPathology)
def test_devel_run(tmpdir, ds_simple, model_cls, root_path=_PATH_HERE):
    """Sample fast dev run..."""
    dm = PlantPathologyDM(
        path_csv=os.path.join(root_path, "data", "train.csv"),
        path_img_dir=os.path.join(root_path, "data", "train_images"),
        simple=ds_simple,
        batch_size=2,
        split=0.6,
    )
    net = LitResnet(arch='resnet18')
    model = model_cls(model=net)

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )
    dm.setup()
    trainer.fit(model, datamodule=dm)

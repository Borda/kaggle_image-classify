import os

import pytest

from kaggle_plantpathology.data import PlantPathologyDM
from kaggle_plantpathology.models import LitPlantPathology, MultiPlantPathology
from pytorch_lightning import Trainer

from tests import _ROOT_TESTS


@pytest.mark.parametrize("model_cls", [LitPlantPathology, MultiPlantPathology])
def test_create_model(model_cls, net: str = "resnet18"):
    model_cls(model=net)


@pytest.mark.parametrize(
    "ds_simple,model_cls",
    [
        (True, LitPlantPathology),
        (False, MultiPlantPathology),
    ],
)
def test_devel_run(tmpdir, ds_simple, model_cls, root_path=_ROOT_TESTS):
    """Sample fast dev run..."""
    dm = PlantPathologyDM(
        path_csv=os.path.join(root_path, "data_plant-pathology", "train.csv"),
        base_path=os.path.join(root_path, "data_plant-pathology"),
        simple=ds_simple,
        batch_size=2,
        split=0.6,
    )
    dm.setup()
    model = model_cls(model="resnet18", num_classes=dm.num_classes)

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

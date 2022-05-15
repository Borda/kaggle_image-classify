import os

from pytorch_lightning import Trainer

from kaggle_imgclassif.cassava.data import CassavaDataModule
from kaggle_imgclassif.cassava.models import LitCassava

from tests import _ROOT_DATA


PATH_DATA = os.path.join(_ROOT_DATA, "cassava")


def test_devel_run(tmpdir, path_data=PATH_DATA):
    """Sample fast dev run..."""
    dm = CassavaDataModule(
        path_csv=os.path.join(path_data, "train.csv"),
        path_img_dir=os.path.join(path_data, "train_images"),
        batch_size=1,
        split=0.6,
    )
    model = LitCassava(model="resnet18")

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        gpus=0,
    )
    dm.setup()
    trainer.fit(model, datamodule=dm)

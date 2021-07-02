#!/usr/bin/env python
# coding: utf-8
#
# # Kaggle: Plant Pathology 2021 - FGVC8
#
# > python cli_train.py \
#       --model.model resnet34 \
#       --data.base_path /mnt/69B27B700DDA7D73/Datasets/plant-pathology-2021-640px/
#

import torch
from pytorch_lightning.utilities.cli import LightningCLI

from kaggle_plantpatho.data import PlantPathologyDM
from kaggle_plantpatho.models import MultiPlantPathology

TRAINER_DEFAULTS = dict(
    gpus=1,
    max_epochs=25,
    precision=16,
    accumulate_grad_batches=10,
    val_check_interval=0.5,
    progress_bar_refresh_rate=1,
    weights_summary='top',
    auto_scale_batch_size='binsearch',
)


class TuneFitCLI(LightningCLI):

    def before_fit(self) -> None:
        """Implement to run some code before fit is started"""
        res = self.trainer.tune(**self.fit_kwargs, scale_batch_size_kwargs=dict(max_trials=5))
        self.instantiate_classes()
        torch.cuda.empty_cache()
        self.datamodule.batch_size = int(res['scale_batch_size'] * 0.9)


if __name__ == '__main__':
    cli = TuneFitCLI(
        model_class=MultiPlantPathology,
        datamodule_class=PlantPathologyDM,
        trainer_defaults=TRAINER_DEFAULTS,
        seed_everything_default=42,
    )

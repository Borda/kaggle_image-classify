#!/usr/bin/env python
# coding: utf-8
# > python train.py --model.model resnet34 --data.base_path /mnt/69B27B700DDA7D73/Datasets/plant-pathology-2021-640px/

# # Kaggle: Plant Pathology 2021 - FGVC8

from pytorch_lightning.utilities.cli import LightningCLI

from kaggle_plantpatho.data import PlantPathologyDM
from kaggle_plantpatho.models import MultiPlantPathology

if __name__ == '__main__':
    tdefaults = dict(
        gpus=1,
        max_epochs=35,
        precision=16,
        accumulate_grad_batches=24,
        val_check_interval=0.5,
        progress_bar_refresh_rate=5,
        weights_summary='top',
        auto_scale_batch_size=True,
    )

    cli = LightningCLI(
        model_class=MultiPlantPathology,
        datamodule_class=PlantPathologyDM,
        trainer_defaults=tdefaults,
        seed_everything_default=42,
    )

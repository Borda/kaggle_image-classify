#!/usr/bin/env python
# coding: utf-8

# # Kaggle: Plant Pathology 2021 - FGVC8


from pytorch_lightning.utilities.cli import LightningCLI

from kaggle_plantpatho.data import PlantPathologyDM


from kaggle_plantpatho.models import MultiPlantPathology


if __name__ == '__main__':
    tdefaults = dict(    max_epochs=35,
    precision=16,
    auto_lr_find=True,
    accumulate_grad_batches=24,
    val_check_interval=0.5,
    progress_bar_refresh_rate=1,
    weights_summary='top',)
    cli = LightningCLI(
        model_class=MultiPlantPathology,
                     datamodule_class=PlantPathologyDM,
        trainer_defaults=tdefaults
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


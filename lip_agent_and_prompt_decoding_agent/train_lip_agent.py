import os
import hydra
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from avg_ckpts import ensemble
from datamodule.data_module_CCS import DataModule_CCS
from lightning_CCS import ModelModule_CCS


@hydra.main(version_base="1.3", config_path="configs", config_name="config_CCS_hand")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    if not os.path.exists(os.path.join(cfg.exp_dir, cfg.exp_name)):
        os.makedirs(os.path.join(cfg.exp_dir, cfg.exp_name))

    print()
    print(cfg.exp_name)
    print(cfg.data.dataset)

    checkpoint = ModelCheckpoint(
        monitor="loss_val",
        mode="min",
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name),
        save_last=True,
        filename="{epoch}-{loss_val}",
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # Set modules and trainer
    modelmodule = ModelModule_CCS(cfg)
    datamodule = DataModule_CCS(cfg)
    trainer = Trainer(
        **cfg.trainer,
        #logger=WandbLogger(name=cfg.exp_name, project="auto_avsr"),
        callbacks=callbacks,
        strategy=DDPPlugin(find_unused_parameters=False),
        gpus=[2]
    )

    trainer.fit(model=modelmodule, datamodule=datamodule)
    ensemble(cfg)


if __name__ == "__main__":
    main()

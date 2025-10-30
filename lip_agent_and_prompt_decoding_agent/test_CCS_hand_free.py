import logging
import os

import hydra
import torch

from pytorch_lightning import Trainer

from datamodule.data_module_CCS import DataModule_CCS
from lip_agent_and_prompt_decoding_agent.lightning_CCS_hand_prompt_decoding import ModelModule_CCS_Hand_prompt_decoding


@hydra.main(version_base="1.3", config_path="configs", config_name="config_CCS_hand_free3_test")
def main(cfg):
    # Set modules and trainer

    hand_weight = 4.5
    ctc_weight = 0.5
    print(cfg.exp_name)
    print(cfg.ckpt_path)
    print(cfg.data)

    print("hand_weight", hand_weight)
    print("ctc_weight", ctc_weight)
    print("===================================")

    modelmodule = ModelModule_CCS_Hand_prompt_decoding(cfg, hand_weight=hand_weight, ctc_weight=ctc_weight)
    datamodule = DataModule_CCS(cfg)
    trainer = Trainer(num_nodes=1, gpus=[6])
    # Training and testing

    ckpt_path = os.path.join(cfg.exp_dir, cfg.exp_name, cfg.ckpt_path)
    modelmodule.load_state_dict(
        torch.load(ckpt_path, map_location=lambda storage, loc: storage).get('state_dict'))
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()

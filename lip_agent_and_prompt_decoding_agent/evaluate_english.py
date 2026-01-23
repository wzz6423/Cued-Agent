#!/usr/bin/env python3
"""
Evaluation script for English lip reading model.
Computes WER and CER on validation set.
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer
from datamodule.data_module_CCS import DataModule_CCS
from lightning_CCS import ModelModule_CCS as ModelModule


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print("=" * 60)
    print("English Lip Reading Model Evaluation")
    print("=" * 60)

    # Override settings for evaluation
    # cfg.exp_name = "lrs2_english_eval"  # Commented: use parameter from CLI

    print(f"\nDataset config: {OmegaConf.to_yaml(cfg.data.dataset)}")

    # Initialize model
    print("\nInitializing model...")
    modelmodule = ModelModule(cfg)

    # Load checkpoint
    # Use the correct checkpoint path based on exp_name
    exp_name = cfg.get('exp_name', 'lrs2_english_v1')
    ckpt_path = f"results/{exp_name}/last.ckpt"
    # Fallback to last.ckpt if it doesn't exist
    if not os.path.exists(ckpt_path):
        ckpt_path = f"results/{exp_name}/last.ckpt"
    if not os.path.exists(ckpt_path):
        # Try averaged model
        ckpt_path = f"results/{exp_name}/model_avg_latest.pth"
    if not os.path.exists(ckpt_path):
        # List available files
        import glob
        candidates = glob.glob(f"results/{exp_name}/*.ckpt")
        if candidates:
            ckpt_path = sorted(candidates)[-1]
    if not os.path.exists(ckpt_path):
        # Try to find any checkpoint
        results_dir = f"results/{exp_name}"
        if os.path.exists(results_dir):
            ckpts = [f for f in os.listdir(results_dir) if f.endswith('.ckpt')]
            if ckpts:
                ckpt_path = os.path.join(results_dir, sorted(ckpts)[-1])
            else:
                ckpt_path = f"results/{exp_name}/last.ckpt"
    # Fallback to last.ckpt if it doesn't exist
    if not os.path.exists(ckpt_path):
        ckpt_path = f"results/{exp_name}/last.ckpt"
    if not os.path.exists(ckpt_path):
        # Try averaged model
        ckpt_path = f"results/{exp_name}/model_avg_latest.pth"
    if not os.path.exists(ckpt_path):
        # List available files
        import glob
        candidates = glob.glob(f"results/{exp_name}/*.ckpt")
        if candidates:
            ckpt_path = sorted(candidates)[-1]

    print(f"Loading checkpoint: {ckpt_path}")

    # Use absolute path if provided, otherwise use ckpt_path
ckpt_to_load = ckpt_path
if not os.path.isabs(ckpt_to_load) and not os.path.exists(ckpt_to_load):
    # Try absolute path construction
    abs_path = os.path.join(
        '/home/ubuntu/wzz/Cued-Agent/lip_agent_and_prompt_decoding_agent/results/lrs2_english_attention_only',
        os.path.basename(ckpt_to_load))
    if os.path.exists(abs_path):
        ckpt_to_load = abs_path
        
checkpoint = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)

    # Load state dict
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Filter out mismatched keys (due to vocab size difference if any)
    model_state = modelmodule.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                print(f"  Skipping {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
        else:
            print(f"  Skipping {k}: not in model")

    modelmodule.load_state_dict(filtered_state, strict=False)
    print(f"  Loaded {len(filtered_state)} parameters")

    # Initialize data module
    print("\nInitializing data module...")
    datamodule = DataModule_CCS(cfg)

    # Setup trainer for testing
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        num_nodes=1,
        logger=False,
    )

    # Run evaluation
    print("\n" + "=" * 60)
    print("Running evaluation on validation set...")
    print("=" * 60)

    results = trainer.test(model=modelmodule, datamodule=datamodule)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    if results:
        for key, value in results[0].items():
            print(f"  {key}: {value:.4f}")

    return results


if __name__ == "__main__":
    main()

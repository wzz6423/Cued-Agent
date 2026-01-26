#!/bin/bash
cd /home/ubuntu/wzz/Cued-Agent/lip_agent_and_prompt_decoding_agent
python3 train_lip_agent.py \
  ++trainer.max_epochs=20 \
  ++trainer.log_every_n_steps=50 \
  exp_name=mvlrs_20ep_full \
  pretrained_model_path=null

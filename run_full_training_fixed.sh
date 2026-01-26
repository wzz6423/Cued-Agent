#!/bin/bash
echo "🚀 完整训练 - 144,286个视频"
cd /home/ubuntu/wzz/Cued-Agent
source .venv/bin/activate
cd lip_agent_and_prompt_decoding_agent

# 后台日志清理
(while true; do sleep 3600; find results -name "*.tfevents.*" -mmin +60 -delete 2>/dev/null; done) &
CLEANER=$!

# 训练 - 使用override
python3 train_lip_agent.py \
  trainer.max_epochs=20 \
  trainer.log_every_n_steps=1000 \
  trainer.val_check_interval=1.0 \
  exp_name=mvlrs_full_144k \
  gpus=1

kill $CLEANER 2>/dev/null
echo "✅ 完成"

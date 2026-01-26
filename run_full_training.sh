#!/bin/bash
# 完整训练 - 144K视频数据集

echo "🚀 开始完整训练"
echo "数据: 144,286个视频 (main 48K + pretrain 96K)"
echo "配置: 最小日志防爆磁盘"
echo ""

cd /home/ubuntu/wzz/Cued-Agent
source .venv/bin/activate

cd lip_agent_and_prompt_decoding_agent

# 每隔1小时清理一次日志
(
while true; do
  sleep 3600
  find results/lightning_logs -name "events.out.tfevents.*" -mmin +60 -delete 2>/dev/null
  echo "$(date): 清理1小时前的日志"
done
) &
CLEANER_PID=$!

# 训练
python3 train_lip_agent.py \
  trainer.max_epochs=20 \
  +trainer.log_every_n_steps=1000 \
  +trainer.val_check_interval=1.0 \
  exp_name=mvlrs_full_144k \
  gpus=1

kill $CLEANER_PID 2>/dev/null
echo "✅ 训练完成"

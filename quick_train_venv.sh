#!/bin/bash
# 快速训练 - 使用虚拟环境

echo "🚀 开始快速训练 (144K视频, 5 epochs baseline)"
echo ""

cd /home/ubuntu/wzz/Cued-Agent
source .venv/bin/activate

cd lip_agent_and_prompt_decoding_agent

# 清理旧日志防止爆磁盘
rm -rf results/lightning_logs/version_* 2>/dev/null

# 训练
python3 train_lip_agent.py \
    data.dataset=mvlrs_v1 \
    data.root_dir=../data \
    data.modality=video \
    data.batch_size=12 \
    data.num_workers=4 \
    trainer.max_epochs=5 \
    trainer.log_every_n_steps=500 \
    trainer.val_check_interval=0.5 \
    trainer.limit_train_batches=1000 \
    trainer.limit_val_batches=100 \
    exp_name=mvlrs_full_baseline \
    gpus=1

echo "✅ 训练完成"

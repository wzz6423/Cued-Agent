#!/bin/bash
# 快速训练脚本 - 在144K数据集上训练5个epoch获取baseline

echo "🚀 开始快速训练 (5 epochs on 144K videos)"
echo "目的: 获取baseline checkpoint用于后续完整评估"
echo ""

cd lip_agent_and_prompt_decoding_agent

# 配置最小日志输出
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 检查GPU
python3 -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 训练5个epoch
python3 train_lip_agent.py \
    --config configs/config.yaml \
    data.dataset=mvlrs_v1 \
    data.root_dir=../data \
    data.modality=video \
    data.batch_size=16 \
    data.num_workers=4 \
    trainer.max_epochs=5 \
    trainer.log_every_n_steps=500 \
    trainer.check_val_every_n_epoch=1 \
    trainer.limit_train_batches=1000 \
    trainer.limit_val_batches=100 \
    exp_name=mvlrs_baseline_5epoch \
    gpus=1

echo ""
echo "✅ 训练完成！Checkpoint保存在 results/lightning_logs/"

#!/bin/bash
# 监控训练完成并自动运行评估

log_file="/home/ubuntu/wzz/Cued-Agent/train_20ep.log"
training_complete=false

echo "📋 等待训练完成 (每5分钟检查一次)..."
echo "预计等待时间: 30-50小时"
echo ""

while [ "$training_complete" = false ]; do
  # 检查是否完成（检查最后一行是否包含"Epoch 19"和"100%"）
  if grep -q "Epoch 19.*100%" "$log_file" 2>/dev/null; then
    echo ""
    echo "✅ 训练完成！"
    training_complete=true
    
    # 运行评估
    echo ""
    echo "🚀 启动自动评估流程..."
    python3 /home/ubuntu/wzz/Cued-Agent/inference_and_evaluate.py
    
  else
    # 显示当前进度
    progress=$(tail -1 "$log_file" | grep -o "Epoch.*" | head -c 80)
    if [ -n "$progress" ]; then
      echo "[$(date '+%H:%M:%S')] $progress"
    fi
    
    sleep 300  # 每5分钟检查一次
  fi
done

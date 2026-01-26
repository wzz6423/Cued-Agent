#!/bin/bash
set -e

log_file="/home/ubuntu/wzz/Cued-Agent/train_20ep.log"

echo "🚀 长期训练监控脚本启动"
echo "预计训练时间: 40-60小时"
echo "脚本将定期检查进度，并在完成后自动运行评估"
echo ""

# 记录开始时间
start_time=$(date +%s)
checkpoint_found=false

while true; do
  current_time=$(date +%s)
  elapsed_seconds=$((current_time - start_time))
  elapsed_hours=$((elapsed_seconds / 3600))
  
  # 读取最后进度行
  last_line=$(tail -1 "$log_file" 2>/dev/null || echo "")
  
  # 提取进度信息
  if echo "$last_line" | grep -q "Epoch"; then
    progress=$(echo "$last_line" | grep -o "Epoch [0-9]*:.*" | head -c 120)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已耗时: ${elapsed_hours}h | $progress"
  fi
  
  # 检查是否完成（Epoch 19 且 100%）
  if echo "$last_line" | grep -q "Epoch 19" && echo "$last_line" | grep -q "100%"; then
    echo ""
    echo "✅ 🎉 训练完成！"
    echo "耗时: ${elapsed_hours} 小时"
    echo ""
    
    # 等待日志完全写入
    sleep 5
    
    # 运行评估
    echo "🚀 启动自动评估流程..."
    if python3 /home/ubuntu/wzz/Cued-Agent/inference_and_evaluate.py; then
      echo "✅ 评估完成"
    else
      echo "⚠️  评估有错误，但继续进行"
    fi
    
    # 生成最终总结
    echo ""
    echo "📄 生成最终报告..."
    cat > /home/ubuntu/wzz/Cued-Agent/TRAINING_SUMMARY.txt << SUMMARY
================================================================================
                        训练完成总结
================================================================================

开始时间: $(date -d @$start_time)
完成时间: $(date)
总耗时: ${elapsed_hours} 小时

配置:
  - 数据集: LRS2 MVLRS (97,657 样本)
  - 轮数: 20 epochs
  - Batch Size: 2 (with gradient accumulation)
  - 优化器: AdamW
  - 架构: Conformer E2E ASR (视频模态)

检查点位置:
  $(find /home/ubuntu/wzz/Cued-Agent/lip_agent_and_prompt_decoding_agent/results -name "*.ckpt" | tail -3)

推理结果:
  $(ls -lh /home/ubuntu/wzz/Cued-Agent/inference_result.* 2>/dev/null | awk '{print $9, $5}')

关键指标:
  请见 FINAL_REPORT.md

================================================================================
SUMMARY
    
    echo "✅ 总结已生成: TRAINING_SUMMARY.txt"
    break
  fi
  
  # 检查是否有错误
  if echo "$last_line" | grep -q -i "error\|exception\|out of memory"; then
    echo ""
    echo "❌ 检测到错误:"
    tail -5 "$log_file"
    break
  fi
  
  # 每10分钟检查一次
  sleep 600
done

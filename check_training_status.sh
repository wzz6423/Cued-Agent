#!/bin/bash
# 定期检查训练状态，每1小时执行一次

log_file="/home/ubuntu/wzz/Cued-Agent/train_20ep.log"

while true; do
  echo ""
  echo "==================== 训练进度检查 ===================="
  echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
  
  # 显示最后的progress行
  tail -1 "$log_file" | grep -o "Epoch.*" || echo "等待进度更新..."
  
  # 检查GPU使用
  echo ""
  echo "💻 GPU状态:"
  nvidia-smi -q -d Index,Memory.Used,Memory.Free 2>/dev/null | grep -E "(Index|Used|Free)" || echo "GPU监控不可用"
  
  # 检查进程状态
  echo ""
  echo "⚙️ 进程状态:"
  ps aux | grep "train_lip_agent.py" | grep -v grep | awk '{print "CPU:"$3"% MEM:"$4"%"}' || echo "训练进程未运行"
  
  echo "========================================================="
  
  # 每3600秒（1小时）检查一次
  sleep 3600
done

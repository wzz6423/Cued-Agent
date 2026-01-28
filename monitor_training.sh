#!/bin/bash
# Real-time training monitor

while true; do
  clear
  echo "═══════════════════════════════════════════════════════════"
  echo "🚀 训练进程监控 ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "═══════════════════════════════════════════════════════════"
  echo ""
  
  # Check process
  if ps aux | grep -q "[p]ython3.*train_lip_agent"; then
    echo "✅ 训练进程: 运行中"
    PID=$(pgrep -f train_lip_agent)
    ps -o pid,etime,pcpu,pmem,cmd -p $PID
  else
    echo "❌ 训练进程: 已停止"
  fi
  
  echo ""
  echo "📊 最新进度 (最后5行):"
  tail -5 train_20ep_clean.log | grep -E "(Epoch|it/s|Error)" || echo "暂无进度信息"
  
  echo ""
  echo "按Ctrl+C退出，每30秒刷新一次"
  sleep 30
done

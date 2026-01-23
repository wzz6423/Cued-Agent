#!/bin/bash

# Setup logrotate for training logs to prevent disk overflow

LOGROTATE_CONFIG="/etc/logrotate.d/cued-agent"
OPTIMIZATION_DIR="/home/ubuntu/wzz/Cued-Agent/optimization_results"

# Create local logrotate configuration (doesn't need sudo)
cat > /home/ubuntu/wzz/Cued-Agent/.logrotate.conf << 'LOGCONF'
/home/ubuntu/wzz/Cued-Agent/optimization_results/*.log {
    size 10M
    rotate 2
    compress
    delaycompress
    missingok
    notifempty
}

/home/ubuntu/wzz/Cued-Agent/lip_agent_and_prompt_decoding_agent/results/lightning_logs/**/*.log {
    size 50M
    rotate 1
    compress
    missingok
    notifempty
}
LOGCONF

echo "✓ Created logrotate config at ~/.logrotate.conf"
echo ""
echo "配置说明:"
echo "  - 优化日志: 大小达 10MB 时自动轮转，保留最新 2 个版本"
echo "  - Lightning 日志: 大小达 50MB 时自动轮转，保留最新 1 个版本"
echo ""
echo "手动执行日志轮转:"
echo "  logrotate -f ~/.logrotate.conf"

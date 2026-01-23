#!/bin/bash

# Auto-cleanup daemon for checkpoints during training

while pgrep -f "train_lip_agent.py" > /dev/null; do
    sleep 1800  # Run every 30 minutes
    /home/ubuntu/wzz/Cued-Agent/cleanup_checkpoints.sh
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-cleanup daemon stopped" >> /home/ubuntu/wzz/Cued-Agent/optimization_results/cleanup.log

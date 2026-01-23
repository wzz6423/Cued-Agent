#!/bin/bash

# Monitor disk usage and log file growth during training

OPTIMIZATION_DIR="/home/ubuntu/wzz/Cued-Agent/optimization_results"
MONITOR_LOG="$OPTIMIZATION_DIR/disk_monitor.log"
THRESHOLD_PERCENT=85

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Disk monitoring started" >> "$MONITOR_LOG"

while pgrep -f "train_lip_agent.py" > /dev/null; do
    # Get disk usage percentage
    DISK_USAGE=$(df /home/ubuntu/wzz/Cued-Agent | awk 'NR==2 {print $(NF-1)}' | sed 's/%//')
    
    # Get log file size
    LOG_SIZE=$(du -sh "$OPTIMIZATION_DIR/training.log" 2>/dev/null | awk '{print $1}')
    LIGHTNING_SIZE=$(du -sh "$OPTIMIZATION_DIR/../lip_agent_and_prompt_decoding_agent/results/lightning_logs" 2>/dev/null | awk '{print $1}')
    
    # Check if threshold exceeded
    if [ "$DISK_USAGE" -gt "$THRESHOLD_PERCENT" ]; then
        echo "[$(date '+%H:%M:%S')] ⚠ WARNING: Disk usage at ${DISK_USAGE}% (threshold: ${THRESHOLD_PERCENT}%)" | tee -a "$MONITOR_LOG"
    else
        echo "[$(date '+%H:%M:%S')] ✓ Disk: ${DISK_USAGE}% | training.log: ${LOG_SIZE} | lightning_logs: ${LIGHTNING_SIZE}" >> "$MONITOR_LOG"
    fi
    
    sleep 300  # Check every 5 minutes
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training ended, disk monitoring stopped" >> "$MONITOR_LOG"

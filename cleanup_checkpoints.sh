#!/bin/bash

# Automatic checkpoint cleanup to prevent disk overflow

RESULTS_DIR="/home/ubuntu/wzz/Cued-Agent/lip_agent_and_prompt_decoding_agent/results"
MAX_CHECKPOINTS_PER_RUN=3  # Keep only last N checkpoints per training run
LOG_FILE="/home/ubuntu/wzz/Cued-Agent/optimization_results/cleanup.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint cleanup started" >> "$LOG_FILE"

# For each training run directory
for run_dir in "$RESULTS_DIR"/*; do
    if [ -d "$run_dir" ]; then
        run_name=$(basename "$run_dir")
        
        # Count .ckpt files
        ckpt_count=$(find "$run_dir" -maxdepth 1 -name "*.ckpt" | wc -l)
        
        if [ "$ckpt_count" -gt "$MAX_CHECKPOINTS_PER_RUN" ]; then
            # Sort by modification time (oldest first) and keep only newest N
            to_delete=$((ckpt_count - MAX_CHECKPOINTS_PER_RUN))
            
            echo "[$(date '+%H:%M:%S')] Cleaning $run_name: removing $to_delete old checkpoints" >> "$LOG_FILE"
            
            find "$run_dir" -maxdepth 1 -name "*.ckpt" -type f | \
                sort -k 6 | \
                head -n "$to_delete" | \
                while read ckpt; do
                    size=$(du -h "$ckpt" | awk '{print $1}')
                    rm -f "$ckpt"
                    echo "[$(date '+%H:%M:%S')] Deleted $ckpt ($size)" >> "$LOG_FILE"
                done
        fi
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleanup finished" >> "$LOG_FILE"

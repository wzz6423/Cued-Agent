#!/bin/bash

# Realtime monitoring for pure attention training (mtlalpha=0.0)

LOG_FILE="/home/ubuntu/wzz/Cued-Agent/optimization_results/training.log"
REPORT_FILE="/home/ubuntu/wzz/Cued-Agent/optimization_results/attention_report.log"
EVAL_SCRIPT="/home/ubuntu/wzz/Cued-Agent/lip_agent_and_prompt_decoding_agent/evaluate_english.py"

echo "🔍 Starting monitoring for attention-only training..." >> "$REPORT_FILE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"

last_epoch=-1

while true; do
    # Check if training is still running
    if ! pgrep -f "train_lip_agent.py" > /dev/null; then
        echo "❌ Training process ended" >> "$REPORT_FILE"
        echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
        
        # Run evaluation
        echo "📊 Running evaluation..." >> "$REPORT_FILE"
        cd /home/ubuntu/wzz/Cued-Agent/lip_agent_and_prompt_decoding_agent
        source ../.venv/bin/activate
        python3 evaluate_english.py exp_name=lrs2_english_attention_only >> "$REPORT_FILE" 2>&1
        echo "✓ Evaluation complete" >> "$REPORT_FILE"
        break
    fi
    
    # Extract latest epoch and loss
    current_epoch=$(grep -oP "Epoch \K\d+" "$LOG_FILE" | tail -1)
    
    if [ -n "$current_epoch" ] && [ "$current_epoch" != "$last_epoch" ]; then
        last_epoch=$current_epoch
        timestamp=$(date '+%H:%M:%S')
        
        # Try to extract metrics
        loss=$(grep "loss:" "$LOG_FILE" | tail -1 | grep -oP 'loss: \K[0-9.]+' || echo "N/A")
        
        echo "[$timestamp] Epoch $current_epoch - Loss: $loss" >> "$REPORT_FILE"
        echo "[$timestamp] Epoch $current_epoch completed"
    fi
    
    sleep 300  # Check every 5 minutes
done

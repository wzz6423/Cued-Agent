#!/bin/bash
# 实时备份 last.ckpt 到 backups/（非侵入，不停止训练）
CKPT_DIR="/home/ubuntu/wzz/Cued-Agent/checkpoints"
BACKUP_DIR="$CKPT_DIR/backups"
KEEP=12  # 保留最近12个备份
LAST_MTIME=0
mkdir -p "$BACKUP_DIR"
while true; do
  if [ ! -f "$CKPT_DIR/last.ckpt" ]; then
    sleep 30
    continue
  fi
  MTIME=$(stat -c %Y "$CKPT_DIR/last.ckpt")
  if [ "$MTIME" -ne "$LAST_MTIME" ]; then
    LAST_MTIME=$MTIME
    ts=$(date +%Y%m%d_%H%M%S)
    cp -v "$CKPT_DIR/last.ckpt" "$BACKUP_DIR/last_${ts}.ckpt" >/dev/null 2>&1 && echo "$(date): backup last.ckpt -> last_${ts}.ckpt" >> "$CKPT_DIR/backup.log"
    # 保留最近 N 个
    ls -1t "$BACKUP_DIR"/last_*.ckpt 2>/dev/null | tail -n +$((KEEP+1)) | xargs -r rm -f
  fi
  sleep 60
done

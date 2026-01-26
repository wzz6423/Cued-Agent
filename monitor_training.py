#!/usr/bin/env python3
"""实时监控训练进度"""
import time
import re
import sys
from datetime import datetime

log_file = '/home/ubuntu/wzz/Cued-Agent/train_20ep.log'
last_epoch = -1
last_batch = -1
start_time = time.time()

print("\n" + "="*80)
print("📊 训练监控启动")
print("="*80 + "\n")

try:
    while True:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # 查找最后一行progress信息
        for line in reversed(lines[-500:]):  # 只检查最后500行
            match = re.search(r'Epoch (\d+):\s+(\d+)%\|.*?\|(\d+)/(\d+)', line)
            if match:
                epoch, pct, batch, total = map(int, match.groups())
                elapsed = time.time() - start_time
                
                if epoch != last_epoch or batch != last_batch:
                    progress_bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
                    hours = elapsed / 3600
                    
                    if batch > 0:
                        time_per_batch = elapsed / batch if batch > 0 else 0
                        remaining = time_per_batch * (total * 20 - (epoch * total + batch))
                        remaining_hours = remaining / 3600
                    else:
                        remaining_hours = 0
                    
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Epoch {epoch}/19 | {progress_bar} {pct}% ({batch}/{total}) | "
                          f"已耗: {hours:.1f}h | 预计: {remaining_hours:.1f}h",
                          end='', flush=True)
                    
                    last_epoch = epoch
                    last_batch = batch
                break
        
        # 每30秒检查一次
        time.sleep(30)
        
except KeyboardInterrupt:
    print("\n\n⏹️  监控停止")
except Exception as e:
    print(f"\n❌ 监控错误: {e}")

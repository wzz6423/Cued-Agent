#!/usr/bin/env python3
"""诊断单个batch的处理速度"""
import time
import torch
import sys
sys.path.insert(0, '/home/ubuntu/wzz/Cued-Agent')

print("\n" + "="*80)
print("🔍 性能诊断：数据加载速度测试")
print("="*80)

try:
    from lip_agent_and_prompt_decoding_agent.datamodule.data_module_CCS import DataModule
    
    print("\n1️⃣  初始化数据模块...")
    t0 = time.time()
    
    dm = DataModule(
        lrs_root='/home/ubuntu/wzz/Cued-Agent/data/mvlrs_v1',
        landmark_root='/home/ubuntu/wzz/Cued-Agent/data/LRS2_landmarks',
        num_workers=2,
        batch_size=4,
        shuffle=False
    )
    dm.setup("fit")
    print(f"   ✅ 耗时: {time.time()-t0:.2f}s\n")
    
    print("2️⃣  加载前3个batch...")
    train_loader = dm.train_dataloader()
    times = []
    
    for i, batch in enumerate(train_loader):
        if i >= 3: break
        t = time.time()
        
        # 转到GPU
        if torch.cuda.is_available():
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda()
        
        elapsed = time.time() - t
        times.append(elapsed)
        print(f"   Batch {i+1}: {elapsed:.3f}s")
    
    avg = sum(times) / len(times)
    total_batches = len(train_loader)
    epoch_hours = (avg * total_batches) / 3600
    
    print(f"\n📊 平均batch耗时: {avg:.3f}s")
    print(f"📈 总batch数: {total_batches}")
    print(f"⏳ 单epoch耗时: {epoch_hours:.2f}小时")
    print(f"⏳ 20轮耗时: {epoch_hours * 20:.1f}小时")
    print("\n" + "="*80)
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

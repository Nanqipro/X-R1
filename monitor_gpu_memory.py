#!/usr/bin/env python3
"""
GPU内存监控脚本

实时监控训练过程中的GPU内存使用情况，帮助诊断OOM问题。
"""

import time
import subprocess
import sys
from datetime import datetime

def get_gpu_memory():
    """获取GPU内存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', 
                                '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        return result.stdout.strip().split('\n') if result.stdout else []
    except Exception as e:
        print(f"错误获取GPU信息: {e}")
        return []

def format_memory(memory_mb):
    """格式化内存显示"""
    if memory_mb >= 1024:
        return f"{memory_mb/1024:.1f}GB"
    else:
        return f"{memory_mb}MB"

def monitor_gpu_memory(interval=5, log_file=None):
    """
    监控GPU内存使用
    
    Parameters
    ----------
    interval : int
        监控间隔（秒）
    log_file : str, optional
        日志文件路径
    """
    print("="*80)
    print("X-R1 GPU内存监控器")
    print("="*80)
    print(f"监控间隔: {interval}秒")
    print("按 Ctrl+C 停止监控")
    print("="*80)
    
    log_fp = None
    if log_file:
        log_fp = open(log_file, 'w', encoding='utf-8')
        log_fp.write("时间,GPU,名称,总内存(GB),已用内存(GB),空闲内存(GB),利用率(%)\n")
    
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            gpu_info = get_gpu_memory()
            
            print(f"\n[{timestamp}] GPU内存状态:")
            print("-" * 80)
            print(f"{'GPU':<4} {'名称':<15} {'总内存':<8} {'已用':<8} {'空闲':<8} {'利用率':<6}")
            print("-" * 80)
            
            for line in gpu_info:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        gpu_idx = parts[0]
                        gpu_name = parts[1][:15] + "..." if len(parts[1]) > 15 else parts[1]
                        total_mem = int(parts[2])
                        used_mem = int(parts[3])
                        free_mem = int(parts[4])
                        gpu_util = parts[5]
                        
                        # 检查内存使用率
                        usage_percent = (used_mem / total_mem) * 100
                        
                        # 颜色编码：绿色<70%, 黄色70-90%, 红色>90%
                        if usage_percent < 70:
                            color = "\033[92m"  # 绿色
                        elif usage_percent < 90:
                            color = "\033[93m"  # 黄色
                        else:
                            color = "\033[91m"  # 红色
                        
                        reset_color = "\033[0m"
                        
                        print(f"{color}GPU{gpu_idx:<3} {gpu_name:<15} {format_memory(total_mem):<8} "
                              f"{format_memory(used_mem):<8} {format_memory(free_mem):<8} {gpu_util}%{reset_color}")
                        
                        # 写入日志文件
                        if log_fp:
                            log_fp.write(f"{timestamp},GPU{gpu_idx},{gpu_name},{total_mem/1024:.1f},"
                                       f"{used_mem/1024:.1f},{free_mem/1024:.1f},{gpu_util}\n")
                            log_fp.flush()
                        
                        # 内存告警
                        if usage_percent > 95:
                            print(f"\033[91m⚠️  警告: GPU{gpu_idx} 内存使用率过高 ({usage_percent:.1f}%)！\033[0m")
                        elif free_mem < 1024:  # 小于1GB
                            print(f"\033[93m⚠️  注意: GPU{gpu_idx} 剩余内存不足 ({format_memory(free_mem)})！\033[0m")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止。")
    finally:
        if log_fp:
            log_fp.close()
            print(f"监控日志已保存到: {log_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="监控GPU内存使用情况")
    parser.add_argument("--interval", "-i", type=int, default=5, 
                       help="监控间隔（秒），默认5秒")
    parser.add_argument("--log", "-l", type=str, 
                       help="保存监控日志到文件")
    
    args = parser.parse_args()
    
    monitor_gpu_memory(args.interval, args.log) 
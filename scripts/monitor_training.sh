#!/bin/bash

# 训练监控脚本 - 实时监控GPU内存和训练状态

LOG_FILE="./output/x_r1_3b_lora_advanced_sampling_generated_x_r1_dataset02_stable.log"

echo "=========================================="
echo "X-R1训练监控脚本"
echo "监控日志文件: $LOG_FILE"
echo "=========================================="

# 监控函数
monitor_training() {
    while true; do
        clear
        echo "=== $(date) ==="
        
        # 显示GPU状态
        echo "📊 GPU内存使用情况:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while read line; do
            echo "  GPU $line"
        done
        
        echo ""
        echo "📈 训练进度:"
        
        # 检查训练进程
        if pgrep -f "src/x_r1/grpo.py" > /dev/null; then
            echo "  ✅ 训练进程正在运行"
            
            # 显示最新的训练日志
            if [ -f "$LOG_FILE" ]; then
                echo "  📝 最新日志 (最后10行):"
                tail -n 10 "$LOG_FILE" | sed 's/^/    /'
                
                # 检查错误
                if tail -n 50 "$LOG_FILE" | grep -i "error\|traceback\|exception" > /dev/null; then
                    echo "  ⚠️  检测到错误，请检查日志！"
                fi
                
                # 显示训练步数
                LAST_STEP=$(grep -o "global_step [0-9]*" "$LOG_FILE" | tail -1 | cut -d' ' -f2)
                if [ ! -z "$LAST_STEP" ]; then
                    echo "  🎯 当前步数: $LAST_STEP"
                fi
                
            else
                echo "  📝 日志文件尚未创建"
            fi
        else
            echo "  ❌ 训练进程未运行"
            break
        fi
        
        echo ""
        echo "按 Ctrl+C 退出监控"
        sleep 10
    done
}

# 启动监控
trap 'echo "退出监控..."; exit 0' INT
monitor_training 
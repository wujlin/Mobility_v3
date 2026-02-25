#!/bin/bash
# === 服务器 B 最终极速脚本 (112核专用版) ===
# 特性: 并发50 + 12G内存/线程 + PM2集群联动 + 自动隔离

# === 配置区 ===
BUCKET_DIR="/tmp/buckets_part1_migrated"
QUARANTINE_DIR="/tmp/buckets_part1_migrated/quarantine"

# 假设你的 repo 软连接是存在的，如果不存在请改为 /media/liuzhihang/仓库/...
INPUT_ROOT="/media/liuzhihang/repo/projects/wellspace/GLAN/PHASE1/spatial_temporal_merge"
OUTPUT_ROOT="/media/liuzhihang/repo/projects/wellspace/GLAN_processed"

# 日志存当前目录
LOG_FILE="./migration_problems.log"

# === 内存设置 (关键) ===
# 你的服务器有 768GB 内存。
# 并发 50 * 12GB = 600GB，预留 168GB 给系统和后端，非常安全且高效。
export NODE_OPTIONS="--max-old-space-size=12288"

# 准备工作
mkdir -p "$QUARANTINE_DIR"
touch "$LOG_FILE"

# === 函数: 检查后端健康 (并发极高时，健康检查很重要) ===
wait_for_backend() {
    local fail_count=0
    while true; do
        # 3秒超时，检查后端
        if curl -s --max-time 3 "http://localhost:3001/api/weather/current" > /dev/null; then
            return 0
        else
            echo "⚠️ [后端拥堵] 等待 2秒..."
            sleep 2
            fail_count=$((fail_count+1))
            
            # 如果连续 10 次没反应 (20秒)，尝试重启后端
            if [ $fail_count -ge 10 ]; then
                echo "🔄 [自动维护] 后端响应过慢，触发 PM2 重载..."
                pm2 reload shadow-backend
                sleep 5
                fail_count=0
            fi
        fi
    done
}

# 读取任务
mapfile -t files < <(ls "$BUCKET_DIR"/*_retry.txt 2>/dev/null)
total=${#files[@]}
count=0

echo "=== 🚀 核动力模式启动: 处理 $total 个任务 (并发=50, 内存=12G) ==="

for bf in "${files[@]}"; do
    count=$((count+1))
    if [ ! -f "$bf" ]; then continue; fi

    # 1. 跑之前测一下后端心跳
    wait_for_backend

    stem=$(basename "$bf" "_retry.txt")
    pure_stem=${stem%-sunlight}
    
    # 找源文件
    input_csv=$(find "$INPUT_ROOT" -name "${pure_stem}.csv" -print -quit)
    
    if [ -z "$input_csv" ]; then
        msg="[$count/$total] ❌ 找不到源文件 $pure_stem -> 移入隔离区"
        echo "$msg"
        echo "$msg" >> "$LOG_FILE"
        mv "$bf" "$QUARANTINE_DIR/"
        continue
    fi
    
    rel_path=${input_csv#$INPUT_ROOT}
    target_dir=$(dirname "$OUTPUT_ROOT$rel_path")
    mkdir -p "$target_dir"

    echo "------------------------------------------------------"
    echo "⚡ [$count/$total] 处理: $pure_stem"

    # 2. 执行计算
    # 使用 $(pwd) 确保找到脚本
    # --concurrency 50: 既然后端有 112 个核，前端并发 50 是很安全的
    timeout 1800s node "$(pwd)/batch-mobility-shadow.mjs" \
        --input "$(dirname "$input_csv")" \
        --output "$target_dir" \
        --backend "http://localhost:3001/api/analysis/shadow" \
        --weather "http://localhost:3001/api/weather/current" \
        --canopy "/media/liuzhihang/repo/projects/wellspace/Tree/HKtree_small.tif" \
        --concurrency 50 \
        --buckets-file "$bf" \
        --target-file "$(basename "$input_csv")"

    EXIT_CODE=$?

    # 3. 结果处理
    if [ $EXIT_CODE -eq 0 ]; then
        rm "$bf"
        echo "✅ [完成] $pure_stem"
    else
        echo "❌ [失败] $pure_stem (Code: $EXIT_CODE) -> 移入隔离区"
        echo "$pure_stem (Code: $EXIT_CODE)" >> "$LOG_FILE"
        mv "$bf" "$QUARANTINE_DIR/"
        
        # 如果超时(124)，说明后端可能有部分实例死锁，轻轻重启一下
        if [ $EXIT_CODE -eq 124 ]; then
            echo "🔄 [超时重置] 刷新后端集群状态..."
            pm2 reload shadow-backend
            sleep 5
        fi
    fi
done

echo "=== 所有任务处理完毕 ==="
#!/bin/bash
# === 服务器 B 最终稳健计算脚本 (PM2自动重启 + 隔离机制) ===

# === 配置区 ===
BUCKET_DIR="/tmp/buckets_part1_migrated"
QUARANTINE_DIR="/tmp/buckets_part1_migrated/quarantine"

# 【注意】你的输入路径：请根据你的实际情况确认是 'repo' 还是 '仓库'
# 既然刚才 migrate_tasks.sh 能跑通，这里沿用 'repo' 路径
INPUT_ROOT="/media/liuzhihang/repo/projects/wellspace/GLAN/PHASE1/spatial_temporal_merge"
OUTPUT_ROOT="/media/liuzhihang/repo/projects/wellspace/GLAN_processed"

# 【修复】日志文件改用当前目录，避免因目录名不匹配(repo/仓库)导致创建失败
LOG_FILE="./migration_problems.log"

# 显式设置内存
export NODE_OPTIONS="--max-old-space-size=16384"

# 确保目录存在
mkdir -p "$QUARANTINE_DIR"
touch "$LOG_FILE"

# === 关键函数：检查后端是否活着 ===
wait_for_backend() {
    # 循环检查，直到后端响应
    while true; do
        # 5秒超时去戳一下后端
        if curl -s --max-time 5 "http://localhost:3001/api/weather/current" > /dev/null; then
            return 0
        else
            echo "⚠️ [后端无响应] 等待 5秒..."
            sleep 5
            # 注意：如果后端长时间没反应，下面的循环主逻辑里有重启机制
        fi
    done
}

# 读取任务列表
mapfile -t files < <(ls "$BUCKET_DIR"/*_retry.txt 2>/dev/null)
total=${#files[@]}
count=0

echo "=== 🚀 启动处理: 共 $total 个任务 (并发=4) ==="

for bf in "${files[@]}"; do
    count=$((count+1))
    
    # 再次检查文件是否存在（防止被移走）
    if [ ! -f "$bf" ]; then continue; fi

    # 1. 在跑之前，确保后端是通的！
    wait_for_backend

    stem=$(basename "$bf" "_retry.txt")
    pure_stem=${stem%-sunlight}
    
    # 寻找源文件
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

    # 2. 执行计算 (超时30分钟，并发4)
    # 注意：这里假设你的 .mjs 脚本就在当前目录下，或者请修正绝对路径
    timeout 1800s node "$(pwd)/batch-mobility-shadow.mjs" \
        --input "$(dirname "$input_csv")" \
        --output "$target_dir" \
        --backend "http://localhost:3001/api/analysis/shadow" \
        --weather "http://localhost:3001/api/weather/current" \
        --canopy "/media/liuzhihang/repo/projects/wellspace/Tree/HKtree_small.tif" \
        --concurrency 4 \
        --buckets-file "$bf" \
        --target-file "$(basename "$input_csv")"

    EXIT_CODE=$?

    # 3. 结果判定
    if [ $EXIT_CODE -eq 0 ]; then
        # 成功
        rm "$bf"
        echo "✅ [完成] $pure_stem"
    else
        # 失败/超时 -> 移入隔离区，防止卡死
        echo "❌ [失败/超时] $pure_stem (Code: $EXIT_CODE) -> 移入隔离区"
        echo "$pure_stem (Code: $EXIT_CODE)" >> "$LOG_FILE"
        mv "$bf" "$QUARANTINE_DIR/"
        
        # 强制休息
        sleep 5
        
        # === 核心自愈机制 ===
        # 如果是超时(124)，说明后端可能死锁了，让 PM2 重启它！
        if [ $EXIT_CODE -eq 124 ]; then
            echo "🔄 [超时检测] 后端疑似卡死，正在通过 PM2 重启..."
            pm2 restart shadow-backend
            sleep 10 # 等待重启完成
        fi
    fi
done

echo "=== 所有队列处理完毕 ==="
echo "失败任务已移至: $QUARANTINE_DIR"
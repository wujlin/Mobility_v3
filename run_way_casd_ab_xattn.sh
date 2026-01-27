#!/bin/bash
# ============================================================================
# WAYCASD AB Experiment: Cross-Attention Ablation
# ============================================================================
# 
# 假设：Cross-attention 从 z_enc 提取的信息在误导 Decoder，导致系统性选错。
#
# 诊断证据：
# - 91.9% 的失败样本 gt_rank=2（GT 总是第二选择）
# - 61.5% 的失败是二选一场景
# - 66.7% 选择了远离终点的方向（但模型有 dest_dist 特征！）
# - 70%+ 的情况模型很"自信"地选错（margin > 0.2）
#
# 实验设计：
# - xattn1: 有 cross-attention（baseline，复现 PASTCTX）
# - xattn0: 无 cross-attention（改用 mean-pooled latent）
#
# 预期结果：
# - 如果 xattn0 性能提升 → cross-attention 是问题根源
# - 如果 xattn0 性能下降 → cross-attention 仍有用，问题在别处
# - 如果性能持平 → cross-attention 既不帮助也不伤害
# ============================================================================

set -e

# Configuration
SEED=0
N_EPOCHS=100
BATCH_SIZE=128
LR=2e-4
MAX_WAY_LEN=160
MAX_LEN=160
N_LATENT=64

# Paths (workstation)
DATA_ROOT="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen"
WAY_DATA="${DATA_ROOT}/WAYCASD1_waydata_rustbelt_seed0_strict_v1"
WAY_ROUTES="${WAY_DATA}/W5_way_routes_strict/way_routes_strict_masklen0.npz"
WAY_GRAPH="${WAY_DATA}/W3_way_graph_strict/way_graph.npz"
WAY_FEATURES="${WAY_DATA}/W4_way_features_sem/way_features.npz"

EXP_ROOT="${DATA_ROOT}/WAYCASD_AB_xattn_strict_sem5_rustbelt_seed${SEED}"
mkdir -p "$EXP_ROOT"

# ============================================================================
# Experiment 1: xattn=1 (with cross-attention, has past_context)
# ============================================================================
echo "============================================================"
echo "[1/2] Training xattn=1 (with cross-attention)"
echo "============================================================"

OUT_DIR_1="${EXP_ROOT}/W6_xattn1"
mkdir -p "$OUT_DIR_1"

python -m src.training.train_way_casd_autoencoder \
    --way_routes_npz "$WAY_ROUTES" \
    --way_graph_npz "$WAY_GRAPH" \
    --way_features_npz "$WAY_FEATURES" \
    --out_dir "$OUT_DIR_1" \
    --batch_size $BATCH_SIZE \
    --n_epochs $N_EPOCHS \
    --lr $LR \
    --seed $SEED \
    --max_way_len $MAX_WAY_LEN \
    --max_len $MAX_LEN \
    --n_latent $N_LATENT \
    --decoder_use_dest_dist \
    --decoder_use_cross_attn \
    --decoder_use_past_context \
    --decoder_past_k 8 \
    2>&1 | tee "${OUT_DIR_1}/run.log"

# ============================================================================
# Experiment 2: xattn=0 (without cross-attention, has past_context)
# ============================================================================
echo "============================================================"
echo "[2/2] Training xattn=0 (without cross-attention)"
echo "============================================================"

OUT_DIR_0="${EXP_ROOT}/W6_xattn0"
mkdir -p "$OUT_DIR_0"

python -m src.training.train_way_casd_autoencoder \
    --way_routes_npz "$WAY_ROUTES" \
    --way_graph_npz "$WAY_GRAPH" \
    --way_features_npz "$WAY_FEATURES" \
    --out_dir "$OUT_DIR_0" \
    --batch_size $BATCH_SIZE \
    --n_epochs $N_EPOCHS \
    --lr $LR \
    --seed $SEED \
    --max_way_len $MAX_WAY_LEN \
    --max_len $MAX_LEN \
    --n_latent $N_LATENT \
    --decoder_use_dest_dist \
    --no-decoder_use_cross_attn \
    --decoder_use_past_context \
    --decoder_past_k 8 \
    2>&1 | tee "${OUT_DIR_0}/run.log"

# ============================================================================
# Evaluation: Run zenc_informativeness on both models
# ============================================================================
echo "============================================================"
echo "Running zenc_informativeness diagnostics..."
echo "============================================================"

for variant in xattn1 xattn0; do
    echo "  Evaluating $variant..."
    OUT_DIR="${EXP_ROOT}/W6_${variant}"
    
    python -m src.evaluation.way_casd_zenc_informativeness \
        --way_routes_npz "$WAY_ROUTES" \
        --way_graph_npz "$WAY_GRAPH" \
        --way_features_npz "$WAY_FEATURES" \
        --ae_ckpt "${OUT_DIR}/ckpt_best.pt" \
        --out_json "${OUT_DIR}/zenc_informativeness.json" \
        --n_routes 200 \
        --seed 42 \
        2>&1 | tee "${OUT_DIR}/run_zenc.log"
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "AB Experiment Complete!"
echo "============================================================"
echo ""
echo "Results:"
for variant in xattn1 xattn0; do
    OUT_DIR="${EXP_ROOT}/W6_${variant}"
    if [ -f "${OUT_DIR}/zenc_informativeness.json" ]; then
        echo "  $variant:"
        python3 -c "
import json
with open('${OUT_DIR}/zenc_informativeness.json') as f:
    d = json.load(f)
s = d.get('summary', {})
true_sr = s.get('true', {}).get('success_rate', 'N/A')
shuffle_sr = s.get('shuffle', {}).get('success_rate', 'N/A')
print(f'    true_success_rate: {true_sr}')
print(f'    shuffle_success_rate: {shuffle_sr}')
"
    fi
done

echo ""
echo "Experiment directory: $EXP_ROOT"

#!/bin/bash
# =====================================================================
# Dense-Reward RL Fine-tuning for Way-CASD Decoder
# =====================================================================
# Two-step pipeline:
#   Step 1: Precompute graph hop-distance matrix (BFS, ~minutes)
#   Step 2: RL training with per-step dense reward shaping
#
# Prerequisites:
#   - E2 e100 (StepEmb) checkpoint
#   - Flow xattn+regionseq checkpoint
#   - Way graph / features / routes / regions npz files
# =====================================================================

set -euo pipefail

# ---------- Paths (workstation) ----------
DATA_ROOT="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_GRAPH="${DATA_ROOT}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_ROOT}/W3_way_features/way_features.npz"
WAY_ROUTES="${DATA_ROOT}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_REGIONS="${DATA_ROOT}/region_sweep/way_regions_louvain_res5_seed0.npz"
SPLIT_JSON="${DATA_ROOT}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"

# Checkpoints (relative to project root).
AE_CKPT="${AE_CKPT:-_sync/wsa/pi_verify/20260214_porto_p1_stepemb_cont_e100_s0/ckpt_best.pt}"
FLOW_CKPT="${FLOW_CKPT:-_sync/wsa/pi_verify/20260212_porto_flow_xattn_regionseq_dev10p_s0/ckpt_best.pt}"

# Output.
GRAPH_DIST_NPZ="${DATA_ROOT}/W2_way_graph/graph_dist_bfs.npz"
RUN_TAG="${RUN_TAG:-dense_sched09to03}"
OUT_DIR="_sync/wsa/pi_verify/$(date +%Y%m%d)_porto_rl_${RUN_TAG}_from_e100_s0"
BFS_WORKERS="${BFS_WORKERS:-8}"
BFS_CHUNK_SIZE="${BFS_CHUNK_SIZE:-64}"
# 1=fast save (larger file), 0=compressed save (slower)
BFS_NO_COMPRESS="${BFS_NO_COMPRESS:-1}"
# Decode-time speed knobs for RL training
DECODE_MAX_CANDIDATES="${DECODE_MAX_CANDIDATES:-0}"
DECODE_CANDIDATE_POLICY="${DECODE_CANDIDATE_POLICY:-first}"

# RL training knobs
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_EPOCHS="${N_EPOCHS:-20}"
CE_WEIGHT_START="${CE_WEIGHT_START:-0.9}"
CE_WEIGHT_END="${CE_WEIGHT_END:-0.3}"
DENSE_SHAPING_COEF="${DENSE_SHAPING_COEF:-0.3}"
DENSE_ARRIVAL_BONUS="${DENSE_ARRIVAL_BONUS:-2.0}"
BEST_METRIC="${BEST_METRIC:-val_success}"
AMP_BF16="${AMP_BF16:-1}"

AMP_FLAG=""
if [ "${AMP_BF16}" = "1" ]; then
    AMP_FLAG="--amp_bf16"
fi

# =====================================================================
# Step 1: Precompute graph hop-distance matrix (if not already done)
# =====================================================================
if [ ! -f "${GRAPH_DIST_NPZ}" ]; then
    echo "=== Step 1: Precomputing graph distance matrix ==="
    NO_COMPRESS_FLAG=""
    if [ "${BFS_NO_COMPRESS}" = "1" ]; then
        NO_COMPRESS_FLAG="--no_compress"
    fi
    python tools/precompute_way_graph_dist.py \
        --way_graph_npz "${WAY_GRAPH}" \
        --out_npz "${GRAPH_DIST_NPZ}" \
        --mode full \
        --num_workers "${BFS_WORKERS}" \
        --chunk_size "${BFS_CHUNK_SIZE}" \
        ${NO_COMPRESS_FLAG}
    echo "=== Graph distance matrix saved to ${GRAPH_DIST_NPZ} ==="
else
    echo "=== Step 1: Graph distance matrix already exists, skipping. ==="
fi

# =====================================================================
# Step 2: RL training with dense reward
# =====================================================================
echo "=== Step 2: Dense-reward RL training ==="
python -m src.training.train_way_casd_decoder_rl \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_CKPT}" \
    --flow_ckpt "${FLOW_CKPT}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --split_json "${SPLIT_JSON}" \
    --out_dir "${OUT_DIR}" \
    --latent_source flow \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --n_epochs "${N_EPOCHS}" \
    --lr 1e-5 \
    --seed 0 \
    --device cuda \
    --tz_offset_hours -5.0 \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --decode_max_candidates "${DECODE_MAX_CANDIDATES}" \
    --decode_candidate_policy "${DECODE_CANDIDATE_POLICY}" \
    --temperature 1.0 \
    --anti_loop_penalty 2.0 \
    --anti_loop_penalty_k 4 \
    --dense_reward \
    --graph_dist_npz "${GRAPH_DIST_NPZ}" \
    --dense_shaping_coef "${DENSE_SHAPING_COEF}" \
    --dense_arrival_bonus "${DENSE_ARRIVAL_BONUS}" \
    --ce_weight "${CE_WEIGHT_START}" \
    --ce_weight_start "${CE_WEIGHT_START}" \
    --ce_weight_end "${CE_WEIGHT_END}" \
    --entropy_coef 0.01 \
    --baseline ema \
    --baseline_ema_beta 0.98 \
    --best_metric "${BEST_METRIC}" \
    ${AMP_FLAG} \
    --reward_success 0.0 \
    --reward_dist 0.0 \
    --penalty_len 0.0 \
    --penalty_loop 0.0 \
    --max_grad_norm 1.0 \
    --log_every 20 \
    --save_every 1

echo "=== Done. Output: ${OUT_DIR} ==="

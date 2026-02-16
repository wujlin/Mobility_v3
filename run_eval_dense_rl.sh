#!/bin/bash
# =====================================================================
# Binned Eval for Dense-Reward RL Checkpoint
# =====================================================================
# Two configs:
#   (A) K=1 greedy (true model capability, no search tricks)
#   (B) K=16 dest-select (paper deployment config)
#
# Uses ckpt_last.pt (epoch 10, best val loss epoch)
# =====================================================================

set -euo pipefail

# ---------- Paths (workstation) ----------
DATA_ROOT="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_GRAPH="${DATA_ROOT}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_ROOT}/W3_way_features/way_features.npz"
WAY_ROUTES="${DATA_ROOT}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
SPLIT_JSON="${DATA_ROOT}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"

# Checkpoints.
RL_DIR="_sync/wsa/pi_verify/20260215_porto_rl_dense_from_e100_s0"
AE_CKPT="${RL_DIR}/ckpt_last.pt"
FLOW_CKPT="_sync/wsa/pi_verify/20260212_porto_flow_xattn_regionseq_dev10p_s0/ckpt_best.pt"

# Output directory.
EVAL_DIR="${RL_DIR}/eval"
mkdir -p "${EVAL_DIR}"

# =====================================================================
# (A) K=1 greedy — true model capability
# =====================================================================
echo "=== Eval A: K=1 greedy ==="
python -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_CKPT}" \
    --latent_source flow \
    --flow_ckpt "${FLOW_CKPT}" \
    --n_samples_per_route 1 \
    --sample_select first \
    --shape_scope none \
    --flow_solver_steps 20 \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --n_routes 5000 \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --no_compare_beam \
    --anti_loop_penalty 2.0 \
    --anti_loop_penalty_k 4 \
    --seed 0 \
    --out_json "${EVAL_DIR}/binned_dense_rl_k1_greedy_n5000.json" \
    --out_per_route_json "${EVAL_DIR}/per_route_dense_rl_k1_greedy_n5000.json" \
    2>&1 | tee "${EVAL_DIR}/run_eval_k1.log"

# =====================================================================
# (B) K=16 dest-select — paper deployment config
# =====================================================================
echo "=== Eval B: K=16 dest-select ==="
python -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_CKPT}" \
    --latent_source flow \
    --flow_ckpt "${FLOW_CKPT}" \
    --n_samples_per_route 16 \
    --sample_select dest \
    --shape_scope none \
    --flow_solver_steps 20 \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --n_routes 5000 \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --no_compare_beam \
    --anti_loop_penalty 2.0 \
    --anti_loop_penalty_k 4 \
    --seed 0 \
    --out_json "${EVAL_DIR}/binned_dense_rl_k16_dest_n5000.json" \
    --out_per_route_json "${EVAL_DIR}/per_route_dense_rl_k16_dest_n5000.json" \
    2>&1 | tee "${EVAL_DIR}/run_eval_k16.log"

echo "=== Done. Results in ${EVAL_DIR}/ ==="

#!/usr/bin/env bash
# ============================================================================
# n_latent=8: Pure AE + Flow z 缺失实验补全
# - D5a: Flow z alignment
# - D5b: Pure AE + Flow z (K8/K16, dest_efficient)
# - D5c: E2 + Flow z (K16, dest_efficient) 补缺口
# ============================================================================
#
# 设计目标：
# 1) 实时日志可见（tee 到文件）
# 2) 单步失败不中断整脚本（避免 terminal “直接退出”）
# 3) 口径统一（city_grid_meta、decode candidates、out_per_route_json）
#
# 用法：
#   bash run_nL8_pureAE_flowz_probe.sh
# 可选覆盖：
#   PYTHON_BIN=python N_ROUTES=5000 EVAL_BATCH_SIZE=128 bash run_nL8_pureAE_flowz_probe.sh
# ============================================================================

set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
N_ROUTES="${N_ROUTES:-5000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE_K8="${EVAL_BATCH_SIZE_K8:-${EVAL_BATCH_SIZE}}"
EVAL_BATCH_SIZE_K16="${EVAL_BATCH_SIZE_K16:-192}"
FLOW_ALIGN_BATCH_SIZE="${FLOW_ALIGN_BATCH_SIZE:-512}"
FLOW_ALIGN_NUM_WORKERS="${FLOW_ALIGN_NUM_WORKERS:-16}"
SEED="${SEED:-0}"

# ---- 路径 ----
DATA_ROOT="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen"
WAY_DATA="${DATA_ROOT}/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${WAY_DATA}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${WAY_DATA}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${WAY_DATA}/W3_way_features/way_features.npz"
WAY_REGIONS="${WAY_DATA}/region_sweep/way_regions_louvain_res5_seed0.npz"
REGION_SEQ="${WAY_DATA}/region_seq_res5/region_seq_min3_max160.npz"
SPLIT_JSON="${WAY_DATA}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"
CITY_GRID_META="/home/jinlin/data/geoexplicit_data/porto_taxi/semantic/osm_road_prob_meta.json"

# n_latent=8 checkpoints（来自 20260219 实验）
NL8_DIR="_sync/wsa/pi_verify/20260219_porto_d1_nlatent8_s0"
AE_CKPT="${NL8_DIR}/D1_ae_nL8/ckpt_best.pt"
FLOW_CKPT="${NL8_DIR}/D2_flow_nL8/ckpt_best.pt"
E2_CKPT="${NL8_DIR}/D3_e2_nL8/ckpt_best.pt"

# 输出目录
OUT_DIR="${NL8_DIR}/D5_pureAE_flowz"
mkdir -p "${OUT_DIR}"

FAIL_STEPS=()

run_step() {
  local name="$1"
  local log_file="$2"
  shift 2
  echo "======================================================================"
  echo ">>> [${name}] START"
  echo ">>> Log: ${log_file}"
  echo "======================================================================"
  "$@" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  if [[ ${rc} -ne 0 ]]; then
    echo ">>> [${name}] FAILED (rc=${rc})"
    FAIL_STEPS+=("${name}")
  else
    echo ">>> [${name}] OK"
  fi
  echo
}

check_path() {
  local p="$1"
  if [[ ! -e "${p}" ]]; then
    echo "[WARN] missing: ${p}"
    return 1
  fi
  return 0
}

echo ">>> [Preflight] 路径检查..."
check_path "${WAY_ROUTES}" || true
check_path "${WAY_GRAPH}" || true
check_path "${WAY_FEATURES}" || true
check_path "${WAY_REGIONS}" || true
check_path "${REGION_SEQ}" || true
check_path "${SPLIT_JSON}" || true
check_path "${CITY_GRID_META}" || true
check_path "${AE_CKPT}" || true
check_path "${FLOW_CKPT}" || true
check_path "${E2_CKPT}" || true
echo
echo ">>> [Config] N_ROUTES=${N_ROUTES}, SEED=${SEED}"
echo ">>> [Config] FLOW_ALIGN_BATCH_SIZE=${FLOW_ALIGN_BATCH_SIZE}, FLOW_ALIGN_NUM_WORKERS=${FLOW_ALIGN_NUM_WORKERS}"
echo ">>> [Config] EVAL_BATCH_SIZE_K8=${EVAL_BATCH_SIZE_K8}, EVAL_BATCH_SIZE_K16=${EVAL_BATCH_SIZE_K16}"
echo

# D5a: Flow z alignment probe
run_step "D5a_flow_alignment_nL8" "${OUT_DIR}/run_d5a_flow_alignment.log" \
  env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m tools.flow_z_alignment_probe \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_CKPT}" \
    --flow_ckpt "${FLOW_CKPT}" \
    --split_json "${SPLIT_JSON}" \
    --region_seq_npz "${REGION_SEQ}" \
    --out_json "${OUT_DIR}/flow_z_alignment_nL8_n${N_ROUTES}.json" \
    --n_routes "${N_ROUTES}" \
    --batch_size "${FLOW_ALIGN_BATCH_SIZE}" \
    --num_workers "${FLOW_ALIGN_NUM_WORKERS}" \
    --seed "${SEED}"

# D5b-k8: Pure AE + Flow z
run_step "D5b_pureAE_flowz_k8_dest_efficient" "${OUT_DIR}/run_d5b_k8.log" \
  env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_CKPT}" \
    --flow_ckpt "${FLOW_CKPT}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --latent_source flow \
    --n_samples_per_route 8 \
    --sample_select dest_efficient \
    --no_compare_beam \
    --n_routes "${N_ROUTES}" \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --decode_max_candidates 0 \
    --decode_candidate_policy first \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --city_grid_meta "0=${CITY_GRID_META}" \
    --eval_batch_size "${EVAL_BATCH_SIZE_K8}" \
    --dump_way_seqs \
    --out_json "${OUT_DIR}/binned_pureAE_nL8_k8_dest_efficient_n${N_ROUTES}.json" \
    --out_per_route_json "${OUT_DIR}/per_route_pureAE_nL8_k8_dest_efficient_n${N_ROUTES}.json" \
    --device cuda \
    --seed "${SEED}"

# D5b-k16: Pure AE + Flow z
run_step "D5b_pureAE_flowz_k16_dest_efficient" "${OUT_DIR}/run_d5b_k16.log" \
  env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_CKPT}" \
    --flow_ckpt "${FLOW_CKPT}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --latent_source flow \
    --n_samples_per_route 16 \
    --sample_select dest_efficient \
    --no_compare_beam \
    --n_routes "${N_ROUTES}" \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --decode_max_candidates 0 \
    --decode_candidate_policy first \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --city_grid_meta "0=${CITY_GRID_META}" \
    --eval_batch_size "${EVAL_BATCH_SIZE_K16}" \
    --dump_way_seqs \
    --out_json "${OUT_DIR}/binned_pureAE_nL8_k16_dest_efficient_n${N_ROUTES}.json" \
    --out_per_route_json "${OUT_DIR}/per_route_pureAE_nL8_k16_dest_efficient_n${N_ROUTES}.json" \
    --device cuda \
    --seed "${SEED}"

# D5c-k16: E2 + Flow z（补缺失）
run_step "D5c_e2_flowz_k16_dest_efficient" "${OUT_DIR}/run_d5c_k16.log" \
  env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${E2_CKPT}" \
    --flow_ckpt "${FLOW_CKPT}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --latent_source flow \
    --n_samples_per_route 16 \
    --sample_select dest_efficient \
    --no_compare_beam \
    --n_routes "${N_ROUTES}" \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --decode_max_candidates 0 \
    --decode_candidate_policy first \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --city_grid_meta "0=${CITY_GRID_META}" \
    --eval_batch_size "${EVAL_BATCH_SIZE_K16}" \
    --dump_way_seqs \
    --out_json "${OUT_DIR}/binned_e2_nL8_k16_dest_efficient_n${N_ROUTES}.json" \
    --out_per_route_json "${OUT_DIR}/per_route_e2_nL8_k16_dest_efficient_n${N_ROUTES}.json" \
    --device cuda \
    --seed "${SEED}"

echo "======================================================================"
echo ">>> DONE. 输出目录: ${OUT_DIR}"
echo ">>> 关键文件:"
echo "  - ${OUT_DIR}/flow_z_alignment_nL8_n${N_ROUTES}.json"
echo "  - ${OUT_DIR}/binned_pureAE_nL8_k8_dest_efficient_n${N_ROUTES}.json"
echo "  - ${OUT_DIR}/binned_pureAE_nL8_k16_dest_efficient_n${N_ROUTES}.json"
echo "  - ${OUT_DIR}/binned_e2_nL8_k16_dest_efficient_n${N_ROUTES}.json"
if [[ ${#FAIL_STEPS[@]} -gt 0 ]]; then
  echo ">>> FAILED STEPS: ${FAIL_STEPS[*]}"
else
  echo ">>> ALL STEPS SUCCEEDED."
fi
echo "======================================================================"

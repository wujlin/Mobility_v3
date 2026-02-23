#!/usr/bin/env bash
# ============================================================================
# n_latent=8: P0 覆盖分析 + P1 Noise-AE（高效修正版）
# ============================================================================
#
# 说明：
# - P0 用已有 D5 per_route 做 OD 覆盖分析（零重解码，分钟级）
# - P1 训练 Noise-AE 后复用 D2 flow 做 K16 评估
# - 所有步骤实时日志；单步失败不会导致整脚本退出
#
# 用法：
#   bash run_nL8_coverage_and_noise.sh
# 可选覆盖：
#   AE_BATCH_SIZE=256 AE_EPOCHS=100 EVAL_BATCH_SIZE=192 bash run_nL8_coverage_and_noise.sh
# ============================================================================

set -u

is_true() {
  local v="${1:-0}"
  [[ "${v}" == "1" || "${v}" == "true" || "${v}" == "TRUE" || "${v}" == "yes" || "${v}" == "YES" ]]
}

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"
N_ROUTES="${N_ROUTES:-5000}"

# ---- 执行开关 ----
RUN_P0="${RUN_P0:-1}"                    # 1=跑 P0 覆盖分析
RUN_P1="${RUN_P1:-1}"                    # 1=跑 P1 Noise-AE 全链路
SKIP_EXISTING="${SKIP_EXISTING:-1}"      # 1=若输出已存在则跳过该步骤
NOISE_RESUME_AUTO="${NOISE_RESUME_AUTO:-1}"  # 1=若存在 ckpt_last.pt 自动续训
FAST_MODE="${FAST_MODE:-0}"              # 1=快速筛选（短训练），0=完整训练

# ---- 资源自适应默认值 ----
CPU_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"
DEFAULT_WORKERS=$(( CPU_CORES / 2 ))
if [[ ${DEFAULT_WORKERS} -lt 4 ]]; then DEFAULT_WORKERS=4; fi
if [[ ${DEFAULT_WORKERS} -gt 32 ]]; then DEFAULT_WORKERS=32; fi

: "${AE_BATCH_SIZE:=384}"
: "${AE_NUM_WORKERS:=${DEFAULT_WORKERS}}"
: "${AE_LR:=2e-4}"
: "${AE_WEIGHT_DECAY:=1e-4}"
: "${NOISE_STD:=1.0}"
: "${NOISE_WARMUP_EPOCHS:=20}"
: "${ZENC_DECODE_BATCH_SIZE:=768}"
: "${EVAL_BATCH_SIZE:=256}"

if is_true "${FAST_MODE}"; then
  : "${AE_EPOCHS:=40}"
  : "${AE_EARLY_STOP_PATIENCE:=8}"
  : "${AE_SAVE_EVERY:=2}"
else
  : "${AE_EPOCHS:=100}"
  : "${AE_EARLY_STOP_PATIENCE:=12}"
  : "${AE_SAVE_EVERY:=5}"
fi

# ---- 公共路径 ----
DATA_ROOT="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen"
WAY_DATA="${DATA_ROOT}/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${WAY_DATA}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${WAY_DATA}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${WAY_DATA}/W3_way_features/way_features.npz"
WAY_REGIONS="${WAY_DATA}/region_sweep/way_regions_louvain_res5_seed0.npz"
REGION_SEQ="${WAY_DATA}/region_seq_res5/region_seq_min3_max160.npz"
SPLIT_JSON="${WAY_DATA}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"
CITY_GRID_META="/home/jinlin/data/geoexplicit_data/porto_taxi/semantic/osm_road_prob_meta.json"

NL8_DIR="_sync/wsa/pi_verify/20260219_porto_d1_nlatent8_s0"
AE_CKPT="${NL8_DIR}/D1_ae_nL8/ckpt_best.pt"
FLOW_CKPT="${NL8_DIR}/D2_flow_nL8/ckpt_best.pt"
NL8_D5_PR_K16="${NL8_DIR}/D5_pureAE_flowz/per_route_pureAE_nL8_k16_dest_efficient_n5000.json"
N64_E2_PR_K16="${N64_E2_PR_K16:-_sync/wsa/pi_verify/20260214_porto_p1_stepemb_cont_e100_s0/phaseB_k16_n5000/per_route_waycasd_e2e100_k16_dest_n5000.json}"

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
check_path "${NL8_D5_PR_K16}" || true
if [[ -f "${N64_E2_PR_K16}" ]]; then
  echo "[OK] n64 baseline per_route found: ${N64_E2_PR_K16}"
else
  echo "[INFO] n64 baseline per_route not found, P0 will run only nL8."
fi
echo
echo ">>> [Config] N_ROUTES=${N_ROUTES}, SEED=${SEED}"
echo ">>> [Config] RUN_P0=${RUN_P0}, RUN_P1=${RUN_P1}, FAST_MODE=${FAST_MODE}, SKIP_EXISTING=${SKIP_EXISTING}, NOISE_RESUME_AUTO=${NOISE_RESUME_AUTO}"
echo ">>> [Config] CPU_CORES=${CPU_CORES}, AE_NUM_WORKERS=${AE_NUM_WORKERS}"
echo ">>> [Config] AE_BATCH_SIZE=${AE_BATCH_SIZE}, AE_NUM_WORKERS=${AE_NUM_WORKERS}, AE_EPOCHS=${AE_EPOCHS}"
echo ">>> [Config] NOISE_STD=${NOISE_STD}, NOISE_WARMUP_EPOCHS=${NOISE_WARMUP_EPOCHS}"
echo ">>> [Config] ZENC_DECODE_BATCH_SIZE=${ZENC_DECODE_BATCH_SIZE}, EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo

# ============================================================================
# P0: OD 覆盖/多样性分析 (nL8 Pure AE K16)
# ============================================================================
P0_DIR="${NL8_DIR}/D6_od_coverage"
mkdir -p "${P0_DIR}"
if is_true "${RUN_P0}"; then
  echo "======================================================================"
  echo "[P0] OD Coverage & Diversity — nL8 Pure AE K16"
  echo "  对比目标: nL64 E2 K16 coverage=9.14%, diversity=0.538"
  echo "======================================================================"

  if [[ -f "${N64_E2_PR_K16}" ]]; then
    P0_OUT="${P0_DIR}/od_coverage_nL8_vs_nL64_k16_from_n5000.json"
    if is_true "${SKIP_EXISTING}" && [[ -f "${P0_OUT}" ]]; then
      echo ">>> [P0_od_coverage_nL8_vs_nL64] SKIP (exists): ${P0_OUT}"
    else
      run_step "P0_od_coverage_nL8_vs_nL64" "${P0_DIR}/run_od_coverage_nL8_vs_nL64.log" \
        env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m src.evaluation.od_coverage_diversity_eval \
          --method "nL8_PureAE_Flowz|greedy=${NL8_D5_PR_K16}" \
          --method "nL64_E2_Mainline|greedy=${N64_E2_PR_K16}" \
          --k 16 \
          --min_routes_per_od 3 \
          --jaccard_threshold 0.5 \
          --save_per_od \
          --out_json "${P0_OUT}"
    fi
  else
    P0_OUT="${P0_DIR}/od_coverage_nL8_k16_from_n5000.json"
    if is_true "${SKIP_EXISTING}" && [[ -f "${P0_OUT}" ]]; then
      echo ">>> [P0_od_coverage_nL8_only] SKIP (exists): ${P0_OUT}"
    else
      run_step "P0_od_coverage_nL8_only" "${P0_DIR}/run_od_coverage_nL8_only.log" \
        env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m src.evaluation.od_coverage_diversity_eval \
          --method "nL8_PureAE_Flowz|greedy=${NL8_D5_PR_K16}" \
          --k 16 \
          --min_routes_per_od 3 \
          --jaccard_threshold 0.5 \
          --save_per_od \
          --out_json "${P0_OUT}"
    fi
  fi

  echo "[P0] Done."
  echo ""
else
  echo ">>> [P0] SKIP by RUN_P0=${RUN_P0}"
  echo ""
fi

# ============================================================================
# P1: Latent Noise AE (nL8, σ=1.0, warmup 20 epochs)
# ============================================================================
#
# 设计思路：
#   - l2_per_dim(flow_z, gt_z) = 1.31 for nL8
#   - 设 noise_std = 1.0 (略低于 Flow 误差水平), warmup 20 epochs
#   - 训练 100 epochs，与 D1_ae_nL8 (noise=0) 对比
#   - 训练完成后直接用 D2_flow_nL8 的 Flow ckpt 做 eval（不重训 Flow）
#
# 与 E2 的本质区别：
#   - E2: decoder 见 Flow z → 发现 z 不可靠 → 学会忽略 z → collapse
#   - Noise AE: decoder 见 GT z + isotropic noise → 必须保留所有维度 → 不会 collapse
#
P1_DIR="${NL8_DIR}/D7_noise_ae"
P1_AE_DIR="${P1_DIR}/ae_nL8_noise1p0"
mkdir -p "${P1_AE_DIR}"
if is_true "${RUN_P1}"; then
  echo "======================================================================"
  echo "[P1a] Train Latent Noise AE (nL8, σ=1.0, warmup=20)"
  echo "======================================================================"

  P1A_OUT="${P1_AE_DIR}/report.json"
  if is_true "${SKIP_EXISTING}" && [[ -f "${P1A_OUT}" ]]; then
    echo ">>> [P1a_train_noise_AE_nL8] SKIP (exists): ${P1A_OUT}"
  else
    RESUME_ARGS=()
    if is_true "${NOISE_RESUME_AUTO}" && [[ -f "${P1_AE_DIR}/ckpt_last.pt" ]]; then
      echo ">>> [P1a] resume from: ${P1_AE_DIR}/ckpt_last.pt"
      RESUME_ARGS=(--resume_ckpt "${P1_AE_DIR}/ckpt_last.pt")
    fi

    run_step "P1a_train_noise_AE_nL8" "${P1_AE_DIR}/run_train.log" \
      env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m src.training.train_way_casd_autoencoder \
        --way_routes_npz "${WAY_ROUTES}" \
        --way_graph_npz "${WAY_GRAPH}" \
        --way_features_npz "${WAY_FEATURES}" \
        --out_dir "${P1_AE_DIR}" \
        --split_json "${SPLIT_JSON}" \
        --n_latent 8 \
        --d_model 256 \
        --n_heads 8 \
        --max_way_len 160 \
        --max_len 160 \
        --max_candidates 32 \
        --decoder_use_dest_dist \
        --decoder_use_cross_attn \
        --decoder_use_cand_query \
        --decoder_use_past_context \
        --decoder_past_k 16 \
        --batch_size "${AE_BATCH_SIZE}" \
        --num_workers "${AE_NUM_WORKERS}" \
        --lr "${AE_LR}" \
        --weight_decay "${AE_WEIGHT_DECAY}" \
        --n_epochs "${AE_EPOCHS}" \
        --save_every "${AE_SAVE_EVERY}" \
        --early_stop_patience "${AE_EARLY_STOP_PATIENCE}" \
        --latent_noise_std "${NOISE_STD}" \
        --noise_warmup_epochs "${NOISE_WARMUP_EPOCHS}" \
        --seed "${SEED}" \
        "${RESUME_ARGS[@]}"
  fi

  echo "[P1a] AE training done."
  echo ""

  # ---- P1b: zenc 探针（noise AE 是否保留 z 信息？） ----
  echo "======================================================================"
  echo "[P1b] zenc informativeness probe (noise AE)"
  echo "  预期: T-S 仍 >> 30pp（noise 不导致 collapse）"
  echo "======================================================================"

  P1B_OUT="${P1_DIR}/zenc_info_noiseAE_nL8_n${N_ROUTES}.json"
  if is_true "${SKIP_EXISTING}" && [[ -f "${P1B_OUT}" ]]; then
    echo ">>> [P1b_zenc_noise_AE] SKIP (exists): ${P1B_OUT}"
  else
    run_step "P1b_zenc_noise_AE" "${P1_DIR}/run_zenc_noiseAE.log" \
      env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m src.evaluation.way_casd_zenc_informativeness \
        --way_routes_npz "${WAY_ROUTES}" \
        --way_graph_npz "${WAY_GRAPH}" \
        --way_features_npz "${WAY_FEATURES}" \
        --ae_ckpt "${P1_AE_DIR}/ckpt_best.pt" \
        --split_json "${SPLIT_JSON}" \
        --split_part test \
        --n_routes "${N_ROUTES}" \
        --decode_max_candidates 32 \
        --decode_candidate_policy first \
        --decode_batch_size "${ZENC_DECODE_BATCH_SIZE}" \
        --log_every_batches 10 \
        --out_json "${P1B_OUT}" \
        --seed "${SEED}"
  fi

  echo "[P1b] Done."
  echo ""

  # ---- P1c: Noise AE + Flow z eval (K=16) ----
  echo "======================================================================"
  echo "[P1c] Noise AE + Flow z binned eval (K=16, dest_efficient)"
  echo "  关键: ae_ckpt 指向 noise AE，flow_ckpt 复用 D2（不重训）"
  echo "======================================================================"

  P1C_OUT="${P1_DIR}/binned_noiseAE_nL8_k16_dest_efficient_n${N_ROUTES}.json"
  if is_true "${SKIP_EXISTING}" && [[ -f "${P1C_OUT}" ]]; then
    echo ">>> [P1c_eval_noise_AE_k16_dest_efficient] SKIP (exists): ${P1C_OUT}"
  else
    run_step "P1c_eval_noise_AE_k16_dest_efficient" "${P1_DIR}/run_eval_noiseAE_k16.log" \
      env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m src.evaluation.way_casd_binned_eval \
        --way_routes_npz "${WAY_ROUTES}" \
        --way_graph_npz "${WAY_GRAPH}" \
        --way_features_npz "${WAY_FEATURES}" \
        --ae_ckpt "${P1_AE_DIR}/ckpt_best.pt" \
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
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --dump_way_seqs \
        --out_json "${P1C_OUT}" \
        --out_per_route_json "${P1_DIR}/per_route_noiseAE_nL8_k16_dest_efficient_n${N_ROUTES}.json" \
        --device cuda \
        --seed "${SEED}"
  fi

  echo "[P1c] Done."
  echo ""
else
  echo ">>> [P1] SKIP by RUN_P1=${RUN_P1}"
  echo ""
fi

echo "======================================================================"
echo "ALL DONE. 汇总检查："
echo "  P0 (OD coverage):    ${P0_DIR}/od_coverage_*.json"
echo "  P1a (noise AE):      ${P1_AE_DIR}/report.json"
echo "  P1b (zenc probe):    ${P1_DIR}/zenc_info_noiseAE_nL8_n${N_ROUTES}.json"
echo "  P1c (noise AE eval): ${P1_DIR}/binned_noiseAE_nL8_k16_dest_efficient_n${N_ROUTES}.json"
if [[ ${#FAIL_STEPS[@]} -gt 0 ]]; then
  echo "  FAILED STEPS: ${FAIL_STEPS[*]}"
else
  echo "  ALL STEPS SUCCEEDED."
fi
echo "======================================================================"

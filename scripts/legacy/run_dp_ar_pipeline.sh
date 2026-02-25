#!/usr/bin/env bash
# ========================================
# Decision Point AR (Proposal C) Pipeline
# ========================================
# 
# Run on wsa workstation with:
#   bash run_dp_ar_pipeline.sh
#
# Prerequisites:
#   - paths_graph.npz (training + test)
#   - road_graph.npz

set -euo pipefail

# ----------------------------------------
# Paths (override via env vars)
# ----------------------------------------
# Minimal required:
#   TRAIN_PATHS, TEST_PATHS, ROAD_GRAPH, OUT_BASE
#
# Backward-compatible defaults (partner's /data/SFM layout):
DATA_DIR="${DATA_DIR:-/data/SFM/v3_data}"
OUT_BASE="${OUT_BASE:-/data/SFM/v3_exps/dp_ar}"
TRAIN_PATHS="${TRAIN_PATHS:-${DATA_DIR}/detroit_train_paths_graph.npz}"
TEST_PATHS="${TEST_PATHS:-${DATA_DIR}/detroit_test_paths_graph.npz}"
ROAD_GRAPH="${ROAD_GRAPH:-${DATA_DIR}/detroit_road_graph.npz}"

# Config
MIN_CHOICE_COUNT=2
MIN_OUT_DEGREE=2
BATCH_SIZE=256
N_EPOCHS=50
LR=0.001
MAX_CANDIDATES=32
SEED="${SEED:-42}"
TZ_OFFSET_HOURS="${TZ_OFFSET_HOURS:--5.0}"

# Optional: auto-use the existing icml2026_routegen layout if RAW_ROOT is set.
if [[ -n "${RAW_ROOT:-}" ]] && [[ ! -f "${TRAIN_PATHS}" ]]; then
  EXP_ROOT="${RAW_ROOT%/}/experiments/icml2026_routegen"
  CAND_T3="${EXP_ROOT}/T3_combo_detroit_columbus_seed0"
  if [[ -f "${CAND_T3}/paths_graph_combo.npz" ]] && [[ -f "${CAND_T3}/road_graph_combo.npz" ]]; then
    TRAIN_PATHS="${CAND_T3}/paths_graph_combo.npz"
    TEST_PATHS="${TEST_PATHS:-${TRAIN_PATHS}}"
    ROAD_GRAPH="${CAND_T3}/road_graph_combo.npz"
    OUT_BASE="${OUT_BASE:-${EXP_ROOT}/T5_dp_ar_combo_seed${SEED}}"
  fi
fi

# Optional: allow IN_DATA/PATHS_NPZ/ROAD_NPZ aliases (workstation-friendly).
if [[ -n "${IN_DATA:-}" ]]; then
  CAND_IN="${IN_DATA%/}"
  if [[ -z "${TRAIN_PATHS:-}" ]] || [[ ! -f "${TRAIN_PATHS}" ]]; then
    if [[ -f "${CAND_IN}/paths_graph_combo.npz" ]]; then
      TRAIN_PATHS="${CAND_IN}/paths_graph_combo.npz"
    fi
  fi
  if [[ -z "${ROAD_GRAPH:-}" ]] || [[ ! -f "${ROAD_GRAPH}" ]]; then
    if [[ -f "${CAND_IN}/road_graph_combo.npz" ]]; then
      ROAD_GRAPH="${CAND_IN}/road_graph_combo.npz"
    fi
  fi
fi
if [[ -n "${PATHS_NPZ:-}" ]] && [[ ( -z "${TRAIN_PATHS:-}" ) || ( ! -f "${TRAIN_PATHS}" ) ]]; then
  if [[ -f "${PATHS_NPZ}" ]]; then
    TRAIN_PATHS="${PATHS_NPZ}"
  fi
fi
if [[ -n "${ROAD_NPZ:-}" ]] && [[ ( -z "${ROAD_GRAPH:-}" ) || ( ! -f "${ROAD_GRAPH}" ) ]]; then
  if [[ -f "${ROAD_NPZ}" ]]; then
    ROAD_GRAPH="${ROAD_NPZ}"
  fi
fi
if [[ -z "${TEST_PATHS:-}" ]] || [[ ! -f "${TEST_PATHS}" ]]; then
  TEST_PATHS="${TEST_PATHS:-${TRAIN_PATHS}}"
fi

echo "Resolved Paths:"
echo "  TRAIN_PATHS=${TRAIN_PATHS}"
echo "  TEST_PATHS=${TEST_PATHS}"
echo "  ROAD_GRAPH=${ROAD_GRAPH}"
echo "  OUT_BASE=${OUT_BASE}"
for f in "${TRAIN_PATHS}" "${TEST_PATHS}" "${ROAD_GRAPH}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing file: ${f}" >&2
    echo "Tip: export TRAIN_PATHS/TEST_PATHS/ROAD_GRAPH/OUT_BASE (or RAW_ROOT) then rerun." >&2
    exit 2
  fi
done
mkdir -p "${OUT_BASE}"

echo "======================================"
echo "Step 1: Build Decision Point Graph"
echo "======================================"

python -m src.data.road_graph.build_decision_point_graph \
    --paths_graph_npz "${TRAIN_PATHS}" \
    --road_graph_npz "${ROAD_GRAPH}" \
    --out_dir "${OUT_BASE}/dp_graph" \
    --min_choice_count ${MIN_CHOICE_COUNT} \
    --min_out_degree ${MIN_OUT_DEGREE} \
    --seed ${SEED}

DP_GRAPH="${OUT_BASE}/dp_graph/decision_point_graph.npz"

echo ""
echo "======================================"
echo "Step 2: Train Decision Point AR Model"
echo "======================================"

python -m src.training.train_graph_ar_decision_point \
    --dp_graph_npz "${DP_GRAPH}" \
    --out_dir "${OUT_BASE}/train" \
    --batch_size ${BATCH_SIZE} \
    --n_epochs ${N_EPOCHS} \
    --lr ${LR} \
    --max_candidates ${MAX_CANDIDATES} \
    --seed ${SEED} \
    --tz_offset_hours ${TZ_OFFSET_HOURS} \
    --device cuda

MODEL_PATH="${OUT_BASE}/train/model.pt"

echo ""
echo "======================================"
echo "Step 3: Evaluate on Test Set (Greedy)"
echo "======================================"

python -m src.training.sample_graph_ar_decision_point \
    --model_path "${MODEL_PATH}" \
    --dp_graph_npz "${DP_GRAPH}" \
    --road_graph_npz "${ROAD_GRAPH}" \
    --paths_graph_npz "${TEST_PATHS}" \
    --out_dir "${OUT_BASE}/eval_greedy" \
    --max_dp_steps 30 \
    --top_k 1 \
    --seed ${SEED} \
    --tz_offset_hours ${TZ_OFFSET_HOURS} \
    --device cuda

echo ""
echo "======================================"
echo "Step 4: Evaluate with Top-3 Sampling"
echo "======================================"

python -m src.training.sample_graph_ar_decision_point \
    --model_path "${MODEL_PATH}" \
    --dp_graph_npz "${DP_GRAPH}" \
    --road_graph_npz "${ROAD_GRAPH}" \
    --paths_graph_npz "${TEST_PATHS}" \
    --out_dir "${OUT_BASE}/eval_top3" \
    --max_dp_steps 30 \
    --top_k 3 \
    --temperature 0.8 \
    --seed ${SEED} \
    --tz_offset_hours ${TZ_OFFSET_HOURS} \
    --device cuda

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo "Results in: ${OUT_BASE}"

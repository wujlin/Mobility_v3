#!/usr/bin/env bash
set -euo pipefail

# Phase B Step1: 排雷验证（容量/收敛性）
# - 目标：确认 dt30 下 Diffusion/Physics 的“收缩/走不动”是否主要由未收敛/容量不足导致
# - 产物：data/experiments/{exp_name}/last.pt + *_eval_{quick,mid}/metrics.json (+ samples.npz)
#
# 用法（推荐在 tmux/nohup 中跑）：
#   bash scripts/phase_b_step1_capacity_check.sh
#
# 可选覆盖：
#   PYTHON=python \
#   SEEDS="0 1 2" \
#   HIDDEN_DIM=128 EPOCHS=100 BATCH_TRAIN=512 LR=1e-3 \
#   MAX_BATCHES_QUICK=10 MAX_BATCHES_MID=200 \
#   bash scripts/phase_b_step1_capacity_check.sh

PYTHON="${PYTHON:-python}"

DATA_PATH="${DATA_PATH:-data/processed_dt30/trajectories/shenzhen_trajectories.h5}"
PROCESSED_DIR="${PROCESSED_DIR:-$(dirname "$(dirname "${DATA_PATH}")")}"
NAV_FILE="${NAV_FILE:-${PROCESSED_DIR}/nav_field.npz}"

OBS_LEN="${OBS_LEN:-8}"
PRED_LEN="${PRED_LEN:-12}"
DIFF_STEPS="${DIFF_STEPS:-100}"
PATCH_SIZE="${PATCH_SIZE:-32}"

HIDDEN_DIM="${HIDDEN_DIM:-128}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
BATCH_TRAIN="${BATCH_TRAIN:-512}"
NUM_WORKERS_TRAIN="${NUM_WORKERS_TRAIN:-0}"

K="${K:-20}"
BATCH_EVAL="${BATCH_EVAL:-32}"
NUM_WORKERS_EVAL="${NUM_WORKERS_EVAL:-0}"
MAX_BATCHES_QUICK="${MAX_BATCHES_QUICK:-10}"
MAX_BATCHES_MID="${MAX_BATCHES_MID:-200}"
RUN_MID="${RUN_MID:-1}"

SEEDS="${SEEDS:-0 1 2}"
RUN_SANITY="${RUN_SANITY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

die() { echo "[ERROR] $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

run() {
  echo "+ $*"
  "$@"
}

maybe_run() {
  local sentinel="$1"; shift
  if [[ -f "$sentinel" ]]; then
    if [[ "$SKIP_EXISTING" == "1" ]]; then
      echo "[SKIP] exists: $sentinel"
      return 0
    fi
    die "exists: $sentinel (set SKIP_EXISTING=1 to skip)"
  fi
  run "$@"
}

echo "PYTHON=${PYTHON}"
echo "PROCESSED_DIR=${PROCESSED_DIR}"
echo "DATA_PATH=${DATA_PATH}"
echo "NAV_FILE=${NAV_FILE}"

require_file "${DATA_PATH}"
require_file "${NAV_FILE}"

if [[ "${MAX_BATCHES_MID}" == "0" ]]; then
  RUN_MID=0
fi

run "${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ "${RUN_SANITY}" == "1" ]]; then
  run "${PYTHON}" -m src.utils.sanity_check \
    --data_path "${PROCESSED_DIR}" \
    --strict \
    --expected_dt 30 \
    --dt_require_constant
fi

for SEED in ${SEEDS}; do
  DIFF_EXP="diff_b_dt30_h${HIDDEN_DIM}_b${BATCH_TRAIN}_lr${LR}_e${EPOCHS}_s${SEED}"
  PHY_EXP="physics_b_dt30_h${HIDDEN_DIM}_b${BATCH_TRAIN}_lr${LR}_e${EPOCHS}_s${SEED}"

  # 1) Train
  maybe_run "data/experiments/${DIFF_EXP}/last.pt" \
    "${PYTHON}" -m src.training.train_diffusion \
      --model_type diffusion \
      --data_path "${DATA_PATH}" \
      --split train \
      --exp_name "${DIFF_EXP}" \
      --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
      --hidden_dim "${HIDDEN_DIM}" --diff_steps "${DIFF_STEPS}" \
      --batch_size "${BATCH_TRAIN}" --epochs "${EPOCHS}" --lr "${LR}" \
      --num_workers "${NUM_WORKERS_TRAIN}" --seed "${SEED}"

  maybe_run "data/experiments/${PHY_EXP}/last.pt" \
    "${PYTHON}" -m src.training.train_diffusion \
      --model_type physics \
      --data_path "${DATA_PATH}" \
      --nav_file "${NAV_FILE}" \
      --split train \
      --exp_name "${PHY_EXP}" \
      --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
      --hidden_dim "${HIDDEN_DIM}" --diff_steps "${DIFF_STEPS}" \
      --patch_size "${PATCH_SIZE}" \
      --batch_size "${BATCH_TRAIN}" --epochs "${EPOCHS}" --lr "${LR}" \
      --num_workers "${NUM_WORKERS_TRAIN}" --seed "${SEED}"

  # 2) Eval quick
  maybe_run "data/experiments/${DIFF_EXP}_eval_quick/metrics.json" \
    "${PYTHON}" -m src.training.evaluate \
      --exp_name "${DIFF_EXP}_eval_quick" \
      --model_type diffusion \
      --data_path "${DATA_PATH}" \
      --checkpoint "data/experiments/${DIFF_EXP}/last.pt" \
      --split test \
      --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
      --hidden_dim "${HIDDEN_DIM}" --diff_steps "${DIFF_STEPS}" \
      --batch_size "${BATCH_EVAL}" --max_batches "${MAX_BATCHES_QUICK}" \
      --num_workers "${NUM_WORKERS_EVAL}" \
      --num_samples_per_condition "${K}" --save_samples 200 --seed "${SEED}"

  maybe_run "data/experiments/${PHY_EXP}_eval_quick/metrics.json" \
    "${PYTHON}" -m src.training.evaluate \
      --exp_name "${PHY_EXP}_eval_quick" \
      --model_type physics \
      --data_path "${DATA_PATH}" \
      --checkpoint "data/experiments/${PHY_EXP}/last.pt" \
      --nav_file "${NAV_FILE}" \
      --split test \
      --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
      --hidden_dim "${HIDDEN_DIM}" --diff_steps "${DIFF_STEPS}" \
      --patch_size "${PATCH_SIZE}" \
      --batch_size "${BATCH_EVAL}" --max_batches "${MAX_BATCHES_QUICK}" \
      --num_workers "${NUM_WORKERS_EVAL}" \
      --num_samples_per_condition "${K}" --save_samples 200 --seed "${SEED}"

  # 3) Eval mid (optional)
  if [[ "${RUN_MID}" == "1" ]]; then
    maybe_run "data/experiments/${DIFF_EXP}_eval_mid/metrics.json" \
      "${PYTHON}" -m src.training.evaluate \
        --exp_name "${DIFF_EXP}_eval_mid" \
        --model_type diffusion \
        --data_path "${DATA_PATH}" \
        --checkpoint "data/experiments/${DIFF_EXP}/last.pt" \
        --split test \
        --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
        --hidden_dim "${HIDDEN_DIM}" --diff_steps "${DIFF_STEPS}" \
        --batch_size "${BATCH_EVAL}" --max_batches "${MAX_BATCHES_MID}" \
        --num_workers "${NUM_WORKERS_EVAL}" \
        --num_samples_per_condition "${K}" --save_samples 200 --seed "${SEED}"

    maybe_run "data/experiments/${PHY_EXP}_eval_mid/metrics.json" \
      "${PYTHON}" -m src.training.evaluate \
        --exp_name "${PHY_EXP}_eval_mid" \
        --model_type physics \
        --data_path "${DATA_PATH}" \
        --checkpoint "data/experiments/${PHY_EXP}/last.pt" \
        --nav_file "${NAV_FILE}" \
        --split test \
        --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
        --hidden_dim "${HIDDEN_DIM}" --diff_steps "${DIFF_STEPS}" \
        --patch_size "${PATCH_SIZE}" \
        --batch_size "${BATCH_EVAL}" --max_batches "${MAX_BATCHES_MID}" \
        --num_workers "${NUM_WORKERS_EVAL}" \
        --num_samples_per_condition "${K}" --save_samples 200 --seed "${SEED}"
  else
    echo "[SKIP] mid eval disabled (set RUN_MID=1 and MAX_BATCHES_MID>0 to enable)"
  fi
done

echo "[DONE] Phase B Step1 capacity/convergence check finished."

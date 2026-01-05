#!/usr/bin/env bash
set -euo pipefail

# Phase B: CFG destination guidance（最小可证伪链条）
#
# 目标：
# - 在当前最优的 Residual Physics (Ref) 上，引入 CFG（目的地 dropout + 推理期 guidance）
# - 尝试在不显著破坏 macro 的前提下，进一步改善 FDE/覆盖（micro）
#
# 用法（建议 tmux）：
#   bash scripts/phase_b_cfg_destination_guidance.sh
#
# 常用覆盖：
#   GPUS="0 1" SEED=0 CFG_DROP=0.1 CFG_SCALES="0 1 2" bash scripts/phase_b_cfg_destination_guidance.sh
#
# 说明：
# - 训练期：--cfg_drop_dest_prob>0 才能让 CFG 在推理期生效
# - 推理期：--cfg_scale=0 等价于不做 guidance（作为对照）

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0}"
GPU0="$(echo "${GPUS}" | awk '{print $1}')"

DATA_PATH="${DATA_PATH:-data/processed_dt30/trajectories/shenzhen_trajectories.h5}"
PROCESSED_DIR="${PROCESSED_DIR:-$(dirname "$(dirname "${DATA_PATH}")")}"
NAV_FILE="${NAV_FILE:-${PROCESSED_DIR}/nav_field.npz}"

OBS_LEN="${OBS_LEN:-8}"
PRED_LEN="${PRED_LEN:-12}"
PATCH_SIZE="${PATCH_SIZE:-32}"
NAV_PATCH_CHANNEL2="${NAV_PATCH_CHANNEL2:-speed}"

PRIOR_CKPT="${PRIOR_CKPT:-data/experiments/baseline_b_dt30/last.pt}"

HIDDEN_DIM="${HIDDEN_DIM:-128}"
DIFF_STEPS="${DIFF_STEPS:-100}"
BATCH_TRAIN="${BATCH_TRAIN:-2048}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-3}"
MAX_BATCHES_TRAIN="${MAX_BATCHES_TRAIN:-200}"
NUM_WORKERS_TRAIN="${NUM_WORKERS_TRAIN:-16}"
SEED="${SEED:-0}"
PRED_TYPE="${PRED_TYPE:-eps}"

# CFG
CFG_DROP="${CFG_DROP:-0.1}"
CFG_UNCOND_MODE="${CFG_UNCOND_MODE:-origin}" # origin|zeros
CFG_SCALES="${CFG_SCALES:-0 1 2}"

# Eval
EVAL_MB_FAST="${EVAL_MB_FAST:-50}"
EVAL_MB_CONFIRM="${EVAL_MB_CONFIRM:-200}"
EVAL_BS_FAST="${EVAL_BS_FAST:-512}"
EVAL_BS_CONFIRM="${EVAL_BS_CONFIRM:-256}"
K_CONFIRM="${K_CONFIRM:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

die() { echo "[ERROR] $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
run() { echo "+ $*"; "$@"; }

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
echo "GPU0=${GPU0}"
echo "DATA_PATH=${DATA_PATH}"
echo "NAV_FILE=${NAV_FILE}"
echo "PRIOR_CKPT=${PRIOR_CKPT}"
echo "CFG_DROP=${CFG_DROP} CFG_SCALES=${CFG_SCALES} CFG_UNCOND_MODE=${CFG_UNCOND_MODE}"

require_file "${DATA_PATH}"
require_file "${NAV_FILE}"
require_file "${PRIOR_CKPT}"

mkdir -p logs/cfg

EXP="phys_residual_cfgp${CFG_DROP}_pred${PRED_TYPE}_e${EPOCHS}_mb${MAX_BATCHES_TRAIN}_s${SEED}"
CKPT="data/experiments/${EXP}/last.pt"

echo "============================================================"
echo "[1/2] Train Residual Physics with CFG-dropout"
echo "============================================================"

maybe_run "${CKPT}" \
  env CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONUNBUFFERED=1 \
    "${PYTHON}" -u -m src.training.train_diffusion \
      --model_type physics \
      --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" --split train \
      --prior_checkpoint "${PRIOR_CKPT}" \
      --exp_name "${EXP}" \
      --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
      --hidden_dim "${HIDDEN_DIM}" --diff_steps "${DIFF_STEPS}" \
      --patch_size "${PATCH_SIZE}" --nav_patch_channel2 "${NAV_PATCH_CHANNEL2}" \
      --batch_size "${BATCH_TRAIN}" --epochs "${EPOCHS}" --lr "${LR}" \
      --max_batches "${MAX_BATCHES_TRAIN}" --num_workers "${NUM_WORKERS_TRAIN}" --seed "${SEED}" \
      --pred_type "${PRED_TYPE}" \
      --cfg_drop_dest_prob "${CFG_DROP}" --cfg_uncond_dest_mode "${CFG_UNCOND_MODE}" \
      |& tee "logs/cfg/${EXP}.log"

echo "============================================================"
echo "[2/2] Eval: cfg_scale sweep (val)"
echo "============================================================"

for S in ${CFG_SCALES}; do
  OUT_FAST="data/experiments/${EXP}_val_k1_fast_cfg${S}/metrics.json"
  maybe_run "${OUT_FAST}" \
    env CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONUNBUFFERED=1 \
      "${PYTHON}" -u -m src.training.evaluate \
        --exp_name "${EXP}_val_k1_fast_cfg${S}" \
        --model_type physics \
        --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" \
        --checkpoint "${CKPT}" --prior_checkpoint "${PRIOR_CKPT}" \
        --split val --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
        --batch_size "${EVAL_BS_FAST}" --num_workers 0 --max_batches "${EVAL_MB_FAST}" \
        --num_samples_per_condition 1 --diff_steps 50 --save_samples 0 --seed "${SEED}" \
        --pred_type auto \
        --cfg_scale "${S}" --cfg_uncond_dest_mode "${CFG_UNCOND_MODE}" \
        |& tee "logs/cfg/${EXP}_val_k1_fast_cfg${S}.log"
done

for S in ${CFG_SCALES}; do
  OUT="data/experiments/${EXP}_val_k${K_CONFIRM}_mb${EVAL_MB_CONFIRM}_cfg${S}/metrics.json"
  maybe_run "${OUT}" \
    env CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONUNBUFFERED=1 \
      "${PYTHON}" -u -m src.training.evaluate \
        --exp_name "${EXP}_val_k${K_CONFIRM}_mb${EVAL_MB_CONFIRM}_cfg${S}" \
        --model_type physics \
        --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" \
        --checkpoint "${CKPT}" --prior_checkpoint "${PRIOR_CKPT}" \
        --split val --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
        --batch_size "${EVAL_BS_CONFIRM}" --num_workers 0 --max_batches "${EVAL_MB_CONFIRM}" \
        --num_samples_per_condition "${K_CONFIRM}" --diff_steps "${DIFF_STEPS}" --save_samples 0 --seed "${SEED}" \
        --pred_type auto \
        --cfg_scale "${S}" --cfg_uncond_dest_mode "${CFG_UNCOND_MODE}" \
        |& tee "logs/cfg/${EXP}_val_k${K_CONFIRM}_cfg${S}.log"
done

echo "[DONE] CFG destination guidance sweep finished."


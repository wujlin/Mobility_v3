#!/usr/bin/env bash
set -euo pipefail

# Phase B v1.2: 可执行训练链条（Prior -> Residual Ref -> Residual v1.2）
#
# 设计目标：
# - KISS：只保留对结果最敏感的开关（prior/disp_weight/pred_type）
# - 高效：支持两张 GPU 并行跑 ref vs v1.2（或多 seed）
# - 可复现：所有关键超参写入 exp_name；日志用 tee 实时可见
#
# 用法（建议 tmux 中跑）：
#   bash scripts/phase_b_v12_training_chain.sh
#
# 常用覆盖（双卡并行）：
#   GPUS="0 1" SEEDS="0" \
#   PRIOR_EPOCHS=50 RES_EPOCHS=20 RES_MAX_BATCHES=200 \
#   bash scripts/phase_b_v12_training_chain.sh

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0}"
SEEDS="${SEEDS:-0}"

DATA_PATH="${DATA_PATH:-data/processed_dt30/trajectories/shenzhen_trajectories.h5}"
PROCESSED_DIR="${PROCESSED_DIR:-$(dirname "$(dirname "${DATA_PATH}")")}"
NAV_FILE="${NAV_FILE:-${PROCESSED_DIR}/nav_field.npz}"

OBS_LEN="${OBS_LEN:-8}"
PRED_LEN="${PRED_LEN:-12}"
DIFF_STEPS_TRAIN="${DIFF_STEPS_TRAIN:-100}"

# 0) Prior（deterministic baseline）
PRIOR_HIDDEN_DIM="${PRIOR_HIDDEN_DIM:-128}"
PRIOR_BATCH="${PRIOR_BATCH:-1024}"
PRIOR_EPOCHS="${PRIOR_EPOCHS:-50}"
PRIOR_LR="${PRIOR_LR:-1e-3}"
PRIOR_SEED="${PRIOR_SEED:-1}"
PRIOR_NUM_WORKERS="${PRIOR_NUM_WORKERS:-8}"
PRIOR_DISP_CLIP_MIN="${PRIOR_DISP_CLIP_MIN:-0.5}"
PRIOR_DISP_CLIP_MAX="${PRIOR_DISP_CLIP_MAX:-5.0}"

# 1) Residual Physics（Ref / v1.2）
RES_HIDDEN_DIM="${RES_HIDDEN_DIM:-128}"
RES_BATCH="${RES_BATCH:-2048}"
RES_EPOCHS="${RES_EPOCHS:-20}"
RES_LR="${RES_LR:-1e-3}"
RES_NUM_WORKERS="${RES_NUM_WORKERS:-16}"
RES_MAX_BATCHES="${RES_MAX_BATCHES:-200}"   # fast iteration：每 epoch 只跑 N 个 batch
PATCH_SIZE="${PATCH_SIZE:-32}"
NAV_PATCH_CHANNEL2="${NAV_PATCH_CHANNEL2:-speed}"
PRED_TYPE="${PRED_TYPE:-eps}"               # eps|v

# v1.2：diff_loss 位移加权（clip）
DIFF_DISP_CLIP_MIN="${DIFF_DISP_CLIP_MIN:-0.5}"
DIFF_DISP_CLIP_MAX="${DIFF_DISP_CLIP_MAX:-3.0}"  # 经验：3.0 比 5.0 更稳，过冲更少

# 2) Eval（两阶段：fast -> confirm）
EVAL_FAST_MB="${EVAL_FAST_MB:-50}"
EVAL_CONFIRM_MB="${EVAL_CONFIRM_MB:-200}"
EVAL_BS_FAST="${EVAL_BS_FAST:-512}"
EVAL_BS_CONFIRM="${EVAL_BS_CONFIRM:-256}"
EVAL_K_FAST="${EVAL_K_FAST:-1}"
EVAL_K_CONFIRM="${EVAL_K_CONFIRM:-10}"
EVAL_DIFF_STEPS="${EVAL_DIFF_STEPS:-100}"
EVAL_SAVE_SAMPLES="${EVAL_SAVE_SAMPLES:-200}"

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

gpu0() { echo "${GPUS}" | awk '{print $1}'; }
gpu1() { echo "${GPUS}" | awk '{print $2}'; }

echo "PYTHON=${PYTHON}"
echo "GPUS=${GPUS}"
echo "SEEDS=${SEEDS}"
echo "DATA_PATH=${DATA_PATH}"
echo "NAV_FILE=${NAV_FILE}"
echo "PRED_TYPE=${PRED_TYPE}"

require_file "${DATA_PATH}"
require_file "${NAV_FILE}"

mkdir -p logs/v12

run "${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "============================================================"
echo "[PRIOR] Step0: Train Prior (disp_weight=clip)"
echo "============================================================"

PRIOR_EXP="prior_dt30_dispw_clip${PRIOR_DISP_CLIP_MIN}_${PRIOR_DISP_CLIP_MAX}_h${PRIOR_HIDDEN_DIM}_b${PRIOR_BATCH}_lr${PRIOR_LR}_e${PRIOR_EPOCHS}_s${PRIOR_SEED}"
PRIOR_CKPT="data/experiments/${PRIOR_EXP}/last.pt"

# Prior training（单卡足够）
maybe_run "${PRIOR_CKPT}" \
  env CUDA_VISIBLE_DEVICES="$(gpu0)" PYTHONUNBUFFERED=1 \
    "${PYTHON}" -u -m src.training.train_baseline \
      --exp_name "${PRIOR_EXP}" \
      --data_path "${DATA_PATH}" --split train \
      --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
      --hidden_dim "${PRIOR_HIDDEN_DIM}" \
      --batch_size "${PRIOR_BATCH}" --epochs "${PRIOR_EPOCHS}" --lr "${PRIOR_LR}" \
      --num_workers "${PRIOR_NUM_WORKERS}" --seed "${PRIOR_SEED}" \
      --disp_weight clip --disp_clip_min "${PRIOR_DISP_CLIP_MIN}" --disp_clip_max "${PRIOR_DISP_CLIP_MAX}" \
      |& tee "logs/v12/${PRIOR_EXP}.log"

# Prior eval（fast）
maybe_run "data/experiments/${PRIOR_EXP}_val_k1_fast/metrics.json" \
  env CUDA_VISIBLE_DEVICES="$(gpu0)" PYTHONUNBUFFERED=1 \
    "${PYTHON}" -u -m src.training.evaluate \
      --exp_name "${PRIOR_EXP}_val_k1_fast" \
      --model_type baseline \
      --data_path "${DATA_PATH}" \
      --checkpoint "${PRIOR_CKPT}" \
      --split val --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
      --batch_size "${EVAL_BS_FAST}" --num_workers 0 --max_batches "${EVAL_FAST_MB}" \
      --num_samples_per_condition 1 --diff_steps "${EVAL_DIFF_STEPS}" \
      --save_samples "${EVAL_SAVE_SAMPLES}" --seed "${PRIOR_SEED}" \
      |& tee "logs/v12/${PRIOR_EXP}_val_k1_fast.log"

for SEED in ${SEEDS}; do

  echo "============================================================"
  echo "[SEED ${SEED}] Step1: Train Residual Physics (Ref vs v1.2)"
  echo "============================================================"

  # Ref：不做 diff_loss 位移加权（用于对照）
  REF_EXP="phys_residual_ref_pred${PRED_TYPE}_e${RES_EPOCHS}_mb${RES_MAX_BATCHES}_s${SEED}"
  REF_CKPT="data/experiments/${REF_EXP}/last.pt"

  # v1.2：diff_loss 位移加权（clip）
  V12_EXP="phys_residual_v12_dispwclipC${DIFF_DISP_CLIP_MAX}_pred${PRED_TYPE}_e${RES_EPOCHS}_mb${RES_MAX_BATCHES}_s${SEED}"
  V12_CKPT="data/experiments/${V12_EXP}/last.pt"

  GPU_A="$(gpu0)"
  GPU_B="$(gpu1)"
  if [[ -n "${GPU_B}" ]]; then
    echo "[INFO] two GPUs detected: ref->GPU${GPU_A}, v1.2->GPU${GPU_B}"
  else
    echo "[INFO] single GPU detected: run sequential on GPU${GPU_A}"
  fi

  if [[ -n "${GPU_B}" ]]; then
    maybe_run "${REF_CKPT}" \
      env CUDA_VISIBLE_DEVICES="${GPU_A}" PYTHONUNBUFFERED=1 \
        "${PYTHON}" -u -m src.training.train_diffusion \
          --model_type physics \
          --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" --split train \
          --prior_checkpoint "${PRIOR_CKPT}" \
          --exp_name "${REF_EXP}" \
          --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
          --hidden_dim "${RES_HIDDEN_DIM}" --diff_steps "${DIFF_STEPS_TRAIN}" \
          --patch_size "${PATCH_SIZE}" --nav_patch_channel2 "${NAV_PATCH_CHANNEL2}" \
          --batch_size "${RES_BATCH}" --epochs "${RES_EPOCHS}" --lr "${RES_LR}" \
          --max_batches "${RES_MAX_BATCHES}" --num_workers "${RES_NUM_WORKERS}" --seed "${SEED}" \
          --pred_type "${PRED_TYPE}" \
          |& tee "logs/v12/${REF_EXP}.log" &

    maybe_run "${V12_CKPT}" \
      env CUDA_VISIBLE_DEVICES="${GPU_B}" PYTHONUNBUFFERED=1 \
        "${PYTHON}" -u -m src.training.train_diffusion \
          --model_type physics \
          --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" --split train \
          --prior_checkpoint "${PRIOR_CKPT}" \
          --exp_name "${V12_EXP}" \
          --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
          --hidden_dim "${RES_HIDDEN_DIM}" --diff_steps "${DIFF_STEPS_TRAIN}" \
          --patch_size "${PATCH_SIZE}" --nav_patch_channel2 "${NAV_PATCH_CHANNEL2}" \
          --batch_size "${RES_BATCH}" --epochs "${RES_EPOCHS}" --lr "${RES_LR}" \
          --max_batches "${RES_MAX_BATCHES}" --num_workers "${RES_NUM_WORKERS}" --seed "${SEED}" \
          --pred_type "${PRED_TYPE}" \
          --diff_disp_weight clip --diff_disp_clip_min "${DIFF_DISP_CLIP_MIN}" --diff_disp_clip_max "${DIFF_DISP_CLIP_MAX}" \
          |& tee "logs/v12/${V12_EXP}.log" &
    wait
  else
    maybe_run "${REF_CKPT}" \
      env CUDA_VISIBLE_DEVICES="${GPU_A}" PYTHONUNBUFFERED=1 \
        "${PYTHON}" -u -m src.training.train_diffusion \
          --model_type physics \
          --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" --split train \
          --prior_checkpoint "${PRIOR_CKPT}" \
          --exp_name "${REF_EXP}" \
          --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
          --hidden_dim "${RES_HIDDEN_DIM}" --diff_steps "${DIFF_STEPS_TRAIN}" \
          --patch_size "${PATCH_SIZE}" --nav_patch_channel2 "${NAV_PATCH_CHANNEL2}" \
          --batch_size "${RES_BATCH}" --epochs "${RES_EPOCHS}" --lr "${RES_LR}" \
          --max_batches "${RES_MAX_BATCHES}" --num_workers "${RES_NUM_WORKERS}" --seed "${SEED}" \
          --pred_type "${PRED_TYPE}" \
          |& tee "logs/v12/${REF_EXP}.log"

    maybe_run "${V12_CKPT}" \
      env CUDA_VISIBLE_DEVICES="${GPU_A}" PYTHONUNBUFFERED=1 \
        "${PYTHON}" -u -m src.training.train_diffusion \
          --model_type physics \
          --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" --split train \
          --prior_checkpoint "${PRIOR_CKPT}" \
          --exp_name "${V12_EXP}" \
          --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
          --hidden_dim "${RES_HIDDEN_DIM}" --diff_steps "${DIFF_STEPS_TRAIN}" \
          --patch_size "${PATCH_SIZE}" --nav_patch_channel2 "${NAV_PATCH_CHANNEL2}" \
          --batch_size "${RES_BATCH}" --epochs "${RES_EPOCHS}" --lr "${RES_LR}" \
          --max_batches "${RES_MAX_BATCHES}" --num_workers "${RES_NUM_WORKERS}" --seed "${SEED}" \
          --pred_type "${PRED_TYPE}" \
          --diff_disp_weight clip --diff_disp_clip_min "${DIFF_DISP_CLIP_MIN}" --diff_disp_clip_max "${DIFF_DISP_CLIP_MAX}" \
          |& tee "logs/v12/${V12_EXP}.log"
  fi

  echo "============================================================"
  echo "[SEED ${SEED}] Step2: Eval (fast -> confirm)"
  echo "============================================================"

  # fast check（val, K=1, ds50）
  maybe_run "data/experiments/${REF_EXP}_val_k1_fast/metrics.json" \
    env CUDA_VISIBLE_DEVICES="$(gpu0)" PYTHONUNBUFFERED=1 \
      "${PYTHON}" -u -m src.training.evaluate \
        --exp_name "${REF_EXP}_val_k1_fast" \
        --model_type physics \
        --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" \
        --checkpoint "${REF_CKPT}" --prior_checkpoint "${PRIOR_CKPT}" \
        --split val --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
        --batch_size "${EVAL_BS_FAST}" --num_workers 0 --max_batches "${EVAL_FAST_MB}" \
        --num_samples_per_condition "${EVAL_K_FAST}" --diff_steps 50 --save_samples 0 --seed "${SEED}" \
        --pred_type auto \
        |& tee "logs/v12/${REF_EXP}_val_k1_fast.log"

  maybe_run "data/experiments/${V12_EXP}_val_k1_fast/metrics.json" \
    env CUDA_VISIBLE_DEVICES="$(gpu0)" PYTHONUNBUFFERED=1 \
      "${PYTHON}" -u -m src.training.evaluate \
        --exp_name "${V12_EXP}_val_k1_fast" \
        --model_type physics \
        --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" \
        --checkpoint "${V12_CKPT}" --prior_checkpoint "${PRIOR_CKPT}" \
        --split val --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
        --batch_size "${EVAL_BS_FAST}" --num_workers 0 --max_batches "${EVAL_FAST_MB}" \
        --num_samples_per_condition "${EVAL_K_FAST}" --diff_steps 50 --save_samples 0 --seed "${SEED}" \
        --pred_type auto \
        |& tee "logs/v12/${V12_EXP}_val_k1_fast.log"

  # confirm（test, K=10, ds100）
  maybe_run "data/experiments/${REF_EXP}_test_k10_confirm/metrics.json" \
    env CUDA_VISIBLE_DEVICES="$(gpu0)" PYTHONUNBUFFERED=1 \
      "${PYTHON}" -u -m src.training.evaluate \
        --exp_name "${REF_EXP}_test_k10_confirm" \
        --model_type physics \
        --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" \
        --checkpoint "${REF_CKPT}" --prior_checkpoint "${PRIOR_CKPT}" \
        --split test --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
        --batch_size "${EVAL_BS_CONFIRM}" --num_workers 0 --max_batches "${EVAL_CONFIRM_MB}" \
        --num_samples_per_condition "${EVAL_K_CONFIRM}" --diff_steps "${EVAL_DIFF_STEPS}" --save_samples 0 --seed "${SEED}" \
        --pred_type auto \
        |& tee "logs/v12/${REF_EXP}_test_k10_confirm.log"

  maybe_run "data/experiments/${V12_EXP}_test_k10_confirm/metrics.json" \
    env CUDA_VISIBLE_DEVICES="$(gpu0)" PYTHONUNBUFFERED=1 \
      "${PYTHON}" -u -m src.training.evaluate \
        --exp_name "${V12_EXP}_test_k10_confirm" \
        --model_type physics \
        --data_path "${DATA_PATH}" --nav_file "${NAV_FILE}" \
        --checkpoint "${V12_CKPT}" --prior_checkpoint "${PRIOR_CKPT}" \
        --split test --obs_len "${OBS_LEN}" --pred_len "${PRED_LEN}" \
        --batch_size "${EVAL_BS_CONFIRM}" --num_workers 0 --max_batches "${EVAL_CONFIRM_MB}" \
        --num_samples_per_condition "${EVAL_K_CONFIRM}" --diff_steps "${EVAL_DIFF_STEPS}" --save_samples 0 --seed "${SEED}" \
        --pred_type auto \
        |& tee "logs/v12/${V12_EXP}_test_k10_confirm.log"

  echo "============================================================"
  echo "[SEED ${SEED}] Summary (test_k10_confirm)"
  echo "============================================================"
  "${PYTHON}" - <<PY
import json
from pathlib import Path
def load(p): return json.loads(Path(p).read_text())
def ratio(m,a,b): return (m.get(a,0.0)/m.get(b,1.0)) if m.get(b,1.0) else 0.0
paths = {
  "Prior(k1,val_fast)": "data/experiments/${PRIOR_EXP}_val_k1_fast/metrics.json",
  "Ref(k10,test)": "data/experiments/${REF_EXP}_test_k10_confirm/metrics.json",
  "v1.2(k10,test)": "data/experiments/${V12_EXP}_test_k10_confirm/metrics.json",
}
print("\\n| Model | ADE | FDE | Spd_R | RoG_R | MSD10_R |")
print("|---|---:|---:|---:|---:|---:|")
for k,p in paths.items():
  if not Path(p).exists():
    continue
  m = load(p)
  print("| {} | {:.2f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} |".format(
    k,
    float(m.get("ADE_best",0.0)),
    float(m.get("FDE_best",0.0)),
    ratio(m,"pred_speed_mean","gt_speed_mean"),
    ratio(m,"Rog","GT_Rog"),
    ratio(m,"MSD_10","GT_MSD_10"),
  ))
PY
done

echo "[DONE] Phase B v1.2 training chain finished."

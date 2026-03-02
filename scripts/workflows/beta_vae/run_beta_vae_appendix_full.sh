#!/usr/bin/env bash

# 全量 appendix 实验一键入口（不省算力版本）
# 顺序：
# 1) beta sweep
# 2) GT μ oracle K-sweep
# 3) Flow 去时间条件消融
# 4) 速度对比表

set -u

echo ">>> [Init] 进入仓库根目录"
PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJ_ROOT" || true

LOG_ROOT="_sync/wsa/pi_verify/20260302_porto_beta_vae_appendix_full_s0"
mkdir -p "${LOG_ROOT}"

FAILED=()

run_job() {
  local name="$1"
  local script="$2"
  echo ""
  echo "======================================================================"
  echo ">>> [${name}] START"
  echo ">>> Script: ${script}"
  echo "======================================================================"
  bash "${script}" 2>&1 | tee "${LOG_ROOT}/${name}.log"
  local rc=${PIPESTATUS[0]}
  if [ "${rc}" -ne 0 ]; then
    echo ">>> [${name}] FAILED (rc=${rc})"
    FAILED+=("${name}")
  else
    echo ">>> [${name}] OK"
  fi
}

run_job "P3_beta_sweep" "scripts/workflows/beta_vae/run_beta_vae_p3_beta_sweep.sh"
run_job "P4_gtmu_k_sweep" "scripts/workflows/beta_vae/run_beta_vae_p4_gtmu_k_sweep.sh"
run_job "P5_flow_notime_ablation" "scripts/workflows/beta_vae/run_beta_vae_p5_flow_notime_ablation.sh"
run_job "P6_speed_benchmark" "scripts/workflows/beta_vae/run_beta_vae_p6_speed_benchmark.sh"

echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
  echo ">>> DONE with failures: ${FAILED[*]}"
else
  echo ">>> DONE: 全量 appendix 实验全部完成"
fi


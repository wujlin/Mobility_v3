#!/usr/bin/env bash

# P2: d=128 ablation 补齐 Phase-C（Coverage / Diversity / MeanMaxJ / CovTauAUC）
# 不训练，只使用现有 per_route 产物。

set -u

echo ">>> [Init] 进入仓库目录"
PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJ_ROOT" || true

BASE128="_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0"
OUT_ROOT="_sync/wsa/pi_verify/20260225_porto_beta_vae128_phasec_s0"
mkdir -p "${OUT_ROOT}"

# 优先使用修复后的 C3(l6) 结果；不存在则回落到 A3
PER_ROUTE_C3="${BASE128}/C3_eval_k16_l6_fix/per_route_betaVAE128_flowmu_l6_k16_dest_efficient_n5000.json"
PER_ROUTE_A3="${BASE128}/A3_eval_k16/per_route_betaVAE128_flowmu_k16_dest_efficient_n5000.json"

FAILED=0

if [ -f "${PER_ROUTE_C3}" ]; then
  PER_ROUTE="${PER_ROUTE_C3}"
  LABEL="BetaVAE128_FlowMu_l6_K16"
elif [ -f "${PER_ROUTE_A3}" ]; then
  PER_ROUTE="${PER_ROUTE_A3}"
  LABEL="BetaVAE128_FlowMu_K16"
else
  echo "[FATAL] per_route 缺失："
  echo "  - ${PER_ROUTE_C3}"
  echo "  - ${PER_ROUTE_A3}"
  FAILED=1
fi

if [ "${FAILED}" -eq 0 ]; then
  echo ">>> [Preflight] 使用输入 per_route: ${PER_ROUTE}"
  ls -lh "${PER_ROUTE}"
fi

if [ "${FAILED}" -eq 0 ]; then
  PYTHONUNBUFFERED=1 conda run -n dpl python -u -m src.evaluation.od_coverage_diversity_eval \
    --method "${LABEL}|greedy=${PER_ROUTE}" \
    --k 16 \
    --min_routes_per_od 3 \
    --jaccard_threshold 0.3 \
    --tau_values "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
    --save_per_od \
    --out_json "${OUT_ROOT}/od_coverage_diversity_betaVAE128_k16_n5000_tau03.json" \
    2>&1 | tee "${OUT_ROOT}/run_od_coverage_diversity_betaVAE128_k16.log"
fi

if [ "${FAILED}" -eq 0 ]; then
  rc=${PIPESTATUS[0]}
  if [ "${rc}" -ne 0 ]; then
    echo ">>> [P2] FAILED (rc=${rc})"
    FAILED=1
  fi
fi

if [ "${FAILED}" -eq 0 ]; then
  echo ""
  echo ">>> [Summary] 提取 P2 关键指标"
  python - <<'PY'
import json
from pathlib import Path
p = Path("_sync/wsa/pi_verify/20260225_porto_beta_vae128_phasec_s0/od_coverage_diversity_betaVAE128_k16_n5000_tau03.json")
d = json.loads(p.read_text())
r = (d.get("summary_table") or [{}])[0]
print("Method | Decode | Arrival | Coverage | Diversity | MeanMaxJ | CovTauAUC | n_OD")
print("------ | ------ | ------- | -------- | --------- | -------- | --------- | ----")
print(
    f"{r.get('method','NA')} | {r.get('decode','NA')} | "
    f"{float(r.get('arrival_rate',0.0)):.4f} | "
    f"{float(r.get('gt_coverage_at_k_mean',0.0)):.4f} | "
    f"{float(r.get('self_diversity_at_k_mean',0.0)):.4f} | "
    f"{float(r.get('mean_max_jaccard_at_k_mean',0.0)):.4f} | "
    f"{float(r.get('coverage_vs_tau_auc',0.0)):.4f} | "
    f"{int(r.get('n_od_groups_kept',0))}"
)
PY
fi

echo ""
if [ "${FAILED}" -eq 0 ]; then
  echo ">>> DONE: P2 d=128 Phase-C 补齐完成"
else
  echo ">>> DONE with failure: P2 d=128 Phase-C"
fi
echo ">>> OUT: ${OUT_ROOT}"

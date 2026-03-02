#!/usr/bin/env bash

# P4: GT μ oracle K sweep (K=4,8,16), eval-only
# 目标：量化 decoder 在“完美 latent 条件”下随 K 的上界变化

set -u

echo ">>> [Init] 进入仓库根目录"
PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJ_ROOT" || true

DATA_BASE="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${DATA_BASE}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${DATA_BASE}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_BASE}/W3_way_features/way_features.npz"
SPLIT_JSON="${DATA_BASE}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"
CITY_META="/home/jinlin/data/geoexplicit_data/porto_taxi/semantic/osm_road_prob_meta.json"

AE_CKPT="_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/A1_beta_vae_ae/ckpt_best.pt"
OUT_ROOT="_sync/wsa/pi_verify/20260302_porto_beta_vae64_gtmu_k_sweep_s0"
mkdir -p "${OUT_ROOT}"

FAILED_STEPS=()

run_step() {
  local name="$1"
  local log_file="$2"
  shift 2
  echo ""
  echo "======================================================================"
  echo ">>> [${name}] START"
  echo ">>> Log: ${log_file}"
  echo "======================================================================"
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  if [ "${rc}" -ne 0 ]; then
    echo ">>> [${name}] FAILED (rc=${rc})"
    FAILED_STEPS+=("${name}")
  else
    echo ">>> [${name}] OK"
  fi
}

echo ">>> [Preflight] 检查关键输入"
for f in "${WAY_ROUTES}" "${WAY_GRAPH}" "${WAY_FEATURES}" "${SPLIT_JSON}" "${CITY_META}" "${AE_CKPT}"; do
  if [ -f "${f}" ]; then
    ls -lh "${f}"
  else
    echo "[MISS] ${f}"
  fi
done

for K in 4 8 16; do
  EVAL_DIR="${OUT_ROOT}/K${K}"
  PHASEC_DIR="${OUT_ROOT}/K${K}_phaseC"
  mkdir -p "${EVAL_DIR}" "${PHASEC_DIR}"

  OJSON="${EVAL_DIR}/binned_betaVAE64_gtmu_k${K}_dest_efficient_antiloop_n5000.json"
  OPER="${EVAL_DIR}/per_route_betaVAE64_gtmu_k${K}_dest_efficient_antiloop_n5000.json"

  if [ ! -f "${OJSON}" ]; then
    run_step "K${K}_eval_gtmu_AL" "${EVAL_DIR}/run_eval_gtmu_k${K}_AL_n5000.log" \
      conda run -n dpl python -u -m src.evaluation.way_casd_binned_eval \
        --way_routes_npz "${WAY_ROUTES}" \
        --way_graph_npz "${WAY_GRAPH}" \
        --way_features_npz "${WAY_FEATURES}" \
        --ae_ckpt "${AE_CKPT}" \
        --latent_source gt \
        --split_json "${SPLIT_JSON}" \
        --split_part test \
        --n_routes 5000 \
        --min_hops 5 \
        --max_way_len 160 \
        --max_decode_len 160 \
        --n_samples_per_route "${K}" \
        --sample_select dest_efficient \
        --decode_max_candidates 0 \
        --decode_candidate_policy first \
        --anti_loop_k 4 \
        --anti_loop_penalty 2.0 \
        --anti_loop_penalty_k 4 \
        --no_compare_beam \
        --city_grid_meta "0=${CITY_META}" \
        --eval_batch_size 256 \
        --dump_way_seqs \
        --out_json "${OJSON}" \
        --out_per_route_json "${OPER}" \
        --device cuda \
        --seed 0
  else
    echo ">>> [K${K}_eval_gtmu_AL] SKIP (binned json exists)"
  fi

  if [ ! -f "${PHASEC_DIR}/od_coverage_diversity_k${K}_tau03_n5000.json" ]; then
    run_step "K${K}_phaseC" "${PHASEC_DIR}/run_od_coverage_diversity_k${K}.log" \
      conda run -n dpl python -u -m src.evaluation.od_coverage_diversity_eval \
        --method "BetaVAE64_GTmu_K${K}_AL|greedy=${OPER}" \
        --k "${K}" \
        --min_routes_per_od 3 \
        --jaccard_threshold 0.3 \
        --tau_values "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
        --save_per_od \
        --out_json "${PHASEC_DIR}/od_coverage_diversity_k${K}_tau03_n5000.json"
  else
    echo ">>> [K${K}_phaseC] SKIP (phaseC json exists)"
  fi
done

echo ""
echo ">>> [Summary] 提取 GT μ oracle K sweep 关键指标"
python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("_sync/wsa/pi_verify/20260302_porto_beta_vae64_gtmu_k_sweep_s0")
ks = [4, 8, 16]
rows = []
for k in ks:
    bp = root / f"K{k}" / f"binned_betaVAE64_gtmu_k{k}_dest_efficient_antiloop_n5000.json"
    cp = root / f"K{k}_phaseC" / f"od_coverage_diversity_k{k}_tau03_n5000.json"
    if (not bp.exists()) or (not cp.exists()):
        rows.append({"k": int(k), "ok": False, "missing": [str(bp), str(cp)]})
        continue
    b = json.loads(bp.read_text(encoding="utf-8")).get("global", {})
    m = json.loads(cp.read_text(encoding="utf-8")).get("methods", [{}])[0]
    rows.append(
        {
            "k": int(k),
            "ok": True,
            "success": float(b.get("success_rate", np.nan)),
            "hit_wall": float(b.get("hit_wall_rate", np.nan)),
            "loop": float(b.get("loop_rate", np.nan)),
            "len_ratio_mean": float(b.get("len_ratio_mean", np.nan)),
            "coverage_tau03": float((m.get("gt_coverage_at_k", {}) or {}).get("mean", np.nan)),
            "diversity": float((m.get("self_diversity_at_k", {}) or {}).get("mean", np.nan)),
            "meanmaxj": float((m.get("mean_max_jaccard_at_k", {}) or {}).get("mean", np.nan)),
            "covtau_auc": float(m.get("coverage_vs_tau_auc", np.nan)),
            "binned_json": str(bp),
            "phasec_json": str(cp),
        }
    )

out = {"ok": True, "task": "beta_vae64_gtmu_k_sweep_summary", "rows": rows}
op = root / "gtmu_k_sweep_summary.json"
op.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] saved: {op}")
print("K | success | hit_wall | loop | len_ratio_mean | cov@0.3 | div | meanmaxj | covtau_auc")
print("-" * 96)
for r in rows:
    if not r.get("ok", False):
        print(f"{r['k']} | MISSING")
        continue
    print(
        f"{r['k']} | {r['success']:.4f} | {r['hit_wall']:.4f} | {r['loop']:.4f} | {r['len_ratio_mean']:.4f} | "
        f"{r['coverage_tau03']:.4f} | {r['diversity']:.4f} | {r['meanmaxj']:.4f} | {r['covtau_auc']:.4f}"
    )
PY

echo ""
if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
  echo ">>> DONE with failures: ${FAILED_STEPS[*]}"
else
  echo ">>> DONE: GT μ oracle K sweep 完成"
fi


#!/usr/bin/env bash

# P1: K sweep（K=1,4,32） for beta-VAE64 + Flow(mu64) + anti-loop
# 输入使用现有 seed0 ckpt，不重训。

set -u

echo ">>> [Init] 进入仓库目录"
PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJ_ROOT" || true

BASE64="_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0"
AE_CKPT="${BASE64}/A1_beta_vae_ae/ckpt_best.pt"
FLOW_CKPT="${BASE64}/A2_flow_on_mu/ckpt_best.pt"
OUT_ROOT="_sync/wsa/pi_verify/20260225_porto_beta_vae64_k_sweep_s0"

DATA_BASE="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${DATA_BASE}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${DATA_BASE}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_BASE}/W3_way_features/way_features.npz"
WAY_REGIONS="${DATA_BASE}/region_sweep/way_regions_louvain_res5_seed0.npz"
SPLIT_JSON="${DATA_BASE}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"
CITY_META="/home/jinlin/data/geoexplicit_data/porto_taxi/semantic/osm_road_prob_meta.json"

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

mkdir -p "${OUT_ROOT}"
echo ">>> [Preflight] 检查关键输入"
for f in "${AE_CKPT}" "${FLOW_CKPT}" "${WAY_ROUTES}" "${WAY_GRAPH}" "${WAY_FEATURES}" "${WAY_REGIONS}" "${SPLIT_JSON}" "${CITY_META}"; do
  if [ -f "${f}" ]; then ls -lh "${f}"; else echo "[MISS] ${f}"; fi
done

for K in 1 4 32; do
  EVAL_DIR="${OUT_ROOT}/K${K}"
  PHASEC_DIR="${OUT_ROOT}/K${K}_phaseC"
  mkdir -p "${EVAL_DIR}" "${PHASEC_DIR}"

  if [ "${K}" -eq 1 ]; then
    NSAMPLES=1
    SSELECT="first"
    OJSON="${EVAL_DIR}/binned_betaVAE64_flowmu_k1_first_antiloop_n5000.json"
    OPER="${EVAL_DIR}/per_route_betaVAE64_flowmu_k1_first_antiloop_n5000.json"
    LOGF="${EVAL_DIR}/run_eval_k1_first_antiloop.log"
  else
    NSAMPLES="${K}"
    SSELECT="dest_efficient"
    OJSON="${EVAL_DIR}/binned_betaVAE64_flowmu_k${K}_dest_efficient_antiloop_n5000.json"
    OPER="${EVAL_DIR}/per_route_betaVAE64_flowmu_k${K}_dest_efficient_antiloop_n5000.json"
    LOGF="${EVAL_DIR}/run_eval_k${K}_dest_efficient_antiloop.log"
  fi

  if [ ! -f "${OJSON}" ]; then
    run_step "K${K}_eval" "${LOGF}" \
      conda run -n dpl python -u -m src.evaluation.way_casd_binned_eval \
        --way_routes_npz "${WAY_ROUTES}" \
        --way_graph_npz "${WAY_GRAPH}" \
        --way_features_npz "${WAY_FEATURES}" \
        --ae_ckpt "${AE_CKPT}" \
        --flow_ckpt "${FLOW_CKPT}" \
        --way_regions_npz "${WAY_REGIONS}" \
        --latent_source flow \
        --split_json "${SPLIT_JSON}" \
        --split_part test \
        --n_routes 5000 \
        --min_hops 5 \
        --max_way_len 160 \
        --max_decode_len 160 \
        --n_samples_per_route "${NSAMPLES}" \
        --sample_select "${SSELECT}" \
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
    echo ">>> [K${K}_eval] SKIP (${OJSON} exists)"
  fi

  if [ ! -f "${PHASEC_DIR}/od_coverage_diversity_k${K}_tau03_n5000.json" ]; then
    run_step "K${K}_phaseC" "${PHASEC_DIR}/run_od_coverage_diversity_k${K}.log" \
      conda run -n dpl python -u -m src.evaluation.od_coverage_diversity_eval \
        --method "BetaVAE64_FlowMu_K${K}_AL|greedy=${OPER}" \
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
echo ">>> [Summary] 提取 K-sweep 核心指标"
python - <<'PY'
import json
from pathlib import Path

def weighted_binned(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    cells = (((d.get("overall") or {}).get("greedy") or {}).get("cells") or {})
    if not cells:
        return None
    n = sum(float(v.get("n", 0) or 0) for v in cells.values())
    if n <= 0:
        return None
    def wavg(k):
        return sum(float(v.get("n", 0) or 0) * float(v.get(k, 0) or 0) for v in cells.values()) / n
    return {
        "success": wavg("success_rate"),
        "hit_wall": wavg("hit_wall_rate"),
        "loop": wavg("loop_rate"),
        "len_ratio_mean": sum(float(v.get("n", 0) or 0) * float(((v.get("len_ratio") or {}).get("mean", 0) or 0)) for v in cells.values()) / n,
    }

def phasec(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    t = d.get("summary_table") or []
    if not t:
        return None
    r = t[0]
    return {
        "coverage": float(r.get("gt_coverage_at_k_mean", 0.0)),
        "diversity": float(r.get("self_diversity_at_k_mean", 0.0)),
        "meanmaxj": float(r.get("mean_max_jaccard_at_k_mean", 0.0)),
        "covtau_auc": float(r.get("coverage_vs_tau_auc", 0.0)),
    }

rows = []
for k in [1,4,32]:
    if k == 1:
        bp = Path(f"_sync/wsa/pi_verify/20260225_porto_beta_vae64_k_sweep_s0/K{k}/binned_betaVAE64_flowmu_k1_first_antiloop_n5000.json")
    else:
        bp = Path(f"_sync/wsa/pi_verify/20260225_porto_beta_vae64_k_sweep_s0/K{k}/binned_betaVAE64_flowmu_k{k}_dest_efficient_antiloop_n5000.json")
    cp = Path(f"_sync/wsa/pi_verify/20260225_porto_beta_vae64_k_sweep_s0/K{k}_phaseC/od_coverage_diversity_k{k}_tau03_n5000.json")
    b = weighted_binned(bp)
    c = phasec(cp)
    row = {"k": k}
    if b: row.update(b)
    if c: row.update(c)
    rows.append(row)

out = {
    "ok": True,
    "task": "beta_vae64_k_sweep_summary",
    "rows": rows,
}
outp = Path("_sync/wsa/pi_verify/20260225_porto_beta_vae64_k_sweep_s0/k_sweep_summary.json")
outp.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] saved: {outp}")
print("K | SR | HitWall | Loop | LenRatio | Coverage | Diversity | MeanMaxJ | CovTauAUC")
print("-" * 90)
for r in rows:
    print(
        f"{r.get('k')} | "
        f"{r.get('success', float('nan')):.4f} | "
        f"{r.get('hit_wall', float('nan')):.4f} | "
        f"{r.get('loop', float('nan')):.4f} | "
        f"{r.get('len_ratio_mean', float('nan')):.4f} | "
        f"{r.get('coverage', float('nan')):.4f} | "
        f"{r.get('diversity', float('nan')):.4f} | "
        f"{r.get('meanmaxj', float('nan')):.4f} | "
        f"{r.get('covtau_auc', float('nan')):.4f}"
    )
PY

echo ""
echo "======================================================================"
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
  echo "DONE with failures: ${FAILED_STEPS[*]}"
else
  echo "DONE: P1 K sweep 完成（K=1/4/32）"
fi
echo "======================================================================"


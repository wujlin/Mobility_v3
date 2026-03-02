#!/usr/bin/env bash

# P5: 时间条件消融（Flow 去掉 hour/dow 条件）
# 口径：复用 beta-VAE64 AE，不重训 AE，仅重训 Flow(mu64) + K16 AL eval + PhaseC

set -u

echo ">>> [Init] 进入仓库根目录"
PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJ_ROOT" || true

DATA_BASE="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${DATA_BASE}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${DATA_BASE}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_BASE}/W3_way_features/way_features.npz"
WAY_REGIONS="${DATA_BASE}/region_sweep/way_regions_louvain_res5_seed0.npz"
REGION_SEQ="${DATA_BASE}/region_seq_res5/region_seq_min3_max160.npz"
SPLIT_JSON="${DATA_BASE}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"
CITY_META="/home/jinlin/data/geoexplicit_data/porto_taxi/semantic/osm_road_prob_meta.json"

AE_CKPT="_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/A1_beta_vae_ae/ckpt_best.pt"
OUT_ROOT="_sync/wsa/pi_verify/20260302_porto_beta_vae64_flow_notime_s0"
OUT_A2="${OUT_ROOT}/A2_flow_on_mu64_notime"
OUT_A3="${OUT_ROOT}/A3_eval_k16_antiloop"
OUT_A4="${OUT_ROOT}/A4_phaseC_covdiv"
mkdir -p "${OUT_A2}" "${OUT_A3}" "${OUT_A4}"

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
for f in "${WAY_ROUTES}" "${WAY_GRAPH}" "${WAY_FEATURES}" "${WAY_REGIONS}" "${REGION_SEQ}" "${SPLIT_JSON}" "${CITY_META}" "${AE_CKPT}"; do
  if [ -f "${f}" ]; then
    ls -lh "${f}"
  else
    echo "[MISS] ${f}"
  fi
done

if [ ! -f "${OUT_A2}/ckpt_best.pt" ]; then
  run_step "A2_train_flow_mu64_notime" "${OUT_A2}/run_train_flow_mu64_notime.log" \
    conda run -n dpl python -u -m src.training.train_way_casd_flow \
      --way_routes_npz "${WAY_ROUTES}" \
      --way_graph_npz "${WAY_GRAPH}" \
      --way_features_npz "${WAY_FEATURES}" \
      --ae_ckpt "${AE_CKPT}" \
      --region_seq_npz "${REGION_SEQ}" \
      --way_regions_npz "${WAY_REGIONS}" \
      --use_region_seq \
      --split_json "${SPLIT_JSON}" \
      --out_dir "${OUT_A2}" \
      --batch_size 512 \
      --num_workers 24 \
      --n_epochs 80 \
      --lr 2e-4 \
      --weight_decay 1e-4 \
      --min_hops 5 \
      --max_way_len 160 \
      --max_candidates 32 \
      --n_layers 6 \
      --flow_target vae_mu \
      --cond_inject xattn \
      --flow_disable_time_cond \
      --save_every 10 \
      --early_stop_patience 15 \
      --device cuda \
      --seed 0
else
  echo ">>> [A2_train_flow_mu64_notime] SKIP (ckpt_best.pt exists)"
fi

if [ ! -f "${OUT_A3}/binned_betaVAE64_flowmu_notime_k16_dest_efficient_antiloop_n5000.json" ]; then
  run_step "A3_eval_k16_AL_notime" "${OUT_A3}/run_eval_k16_AL_notime_n5000.log" \
    conda run -n dpl python -u -m src.evaluation.way_casd_binned_eval \
      --way_routes_npz "${WAY_ROUTES}" \
      --way_graph_npz "${WAY_GRAPH}" \
      --way_features_npz "${WAY_FEATURES}" \
      --ae_ckpt "${AE_CKPT}" \
      --flow_ckpt "${OUT_A2}/ckpt_best.pt" \
      --way_regions_npz "${WAY_REGIONS}" \
      --latent_source flow \
      --split_json "${SPLIT_JSON}" \
      --split_part test \
      --n_routes 5000 \
      --min_hops 5 \
      --max_way_len 160 \
      --max_decode_len 160 \
      --n_samples_per_route 16 \
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
      --out_json "${OUT_A3}/binned_betaVAE64_flowmu_notime_k16_dest_efficient_antiloop_n5000.json" \
      --out_per_route_json "${OUT_A3}/per_route_betaVAE64_flowmu_notime_k16_dest_efficient_antiloop_n5000.json" \
      --device cuda \
      --seed 0
else
  echo ">>> [A3_eval_k16_AL_notime] SKIP (binned json exists)"
fi

if [ ! -f "${OUT_A4}/od_coverage_diversity_betaVAE64_flowmu_notime_k16_AL_n5000_tau03.json" ]; then
  run_step "A4_phaseC_notime" "${OUT_A4}/run_od_coverage_diversity_k16_tau03.log" \
    conda run -n dpl python -u -m src.evaluation.od_coverage_diversity_eval \
      --method "BetaVAE64_FlowMu_NoTime_K16_AL|greedy=${OUT_A3}/per_route_betaVAE64_flowmu_notime_k16_dest_efficient_antiloop_n5000.json" \
      --k 16 \
      --min_routes_per_od 3 \
      --jaccard_threshold 0.3 \
      --tau_values "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
      --save_per_od \
      --out_json "${OUT_A4}/od_coverage_diversity_betaVAE64_flowmu_notime_k16_AL_n5000_tau03.json"
else
  echo ">>> [A4_phaseC_notime] SKIP (phaseC json exists)"
fi

echo ""
echo ">>> [Summary] 提取 No-Time 消融关键指标"
python - <<'PY'
import json
from pathlib import Path
import numpy as np

p_bin = Path("_sync/wsa/pi_verify/20260302_porto_beta_vae64_flow_notime_s0/A3_eval_k16_antiloop/binned_betaVAE64_flowmu_notime_k16_dest_efficient_antiloop_n5000.json")
p_cov = Path("_sync/wsa/pi_verify/20260302_porto_beta_vae64_flow_notime_s0/A4_phaseC_covdiv/od_coverage_diversity_betaVAE64_flowmu_notime_k16_AL_n5000_tau03.json")
out = Path("_sync/wsa/pi_verify/20260302_porto_beta_vae64_flow_notime_s0/notime_ablation_summary.json")

obj = {"ok": False}
if p_bin.exists() and p_cov.exists():
    b = json.loads(p_bin.read_text(encoding="utf-8")).get("global", {})
    m = json.loads(p_cov.read_text(encoding="utf-8")).get("methods", [{}])[0]
    obj = {
        "ok": True,
        "task": "beta_vae64_flow_notime_ablation",
        "success": float(b.get("success_rate", np.nan)),
        "hit_wall": float(b.get("hit_wall_rate", np.nan)),
        "loop": float(b.get("loop_rate", np.nan)),
        "len_ratio_mean": float(b.get("len_ratio_mean", np.nan)),
        "coverage_tau03": float((m.get("gt_coverage_at_k", {}) or {}).get("mean", np.nan)),
        "diversity": float((m.get("self_diversity_at_k", {}) or {}).get("mean", np.nan)),
        "meanmaxj": float((m.get("mean_max_jaccard_at_k", {}) or {}).get("mean", np.nan)),
        "covtau_auc": float(m.get("coverage_vs_tau_auc", np.nan)),
        "binned_json": str(p_bin),
        "phasec_json": str(p_cov),
    }
else:
    obj = {"ok": False, "missing": [str(p_bin), str(p_cov)]}

out.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] saved: {out}")
print(json.dumps(obj, ensure_ascii=False, indent=2))
PY

echo ""
if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
  echo ">>> DONE with failures: ${FAILED_STEPS[*]}"
else
  echo ">>> DONE: No-Time Flow 消融完成"
fi


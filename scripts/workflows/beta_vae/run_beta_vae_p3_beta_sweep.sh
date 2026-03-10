#!/usr/bin/env bash

# P3: beta sweep (d=64 fixed) for beta-VAE + Flow(mu64) + K16 AL + PhaseC
# betas: 0.001 / 0.005 / 0.02 / 0.05
# 设计目标：
# - 不省算力，完整跑通
# - 实时日志（tee）
# - 可断点续跑（已有产物则跳过）

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

OUT_ROOT="_sync/wsa/pi_verify/20260302_porto_beta_vae64_beta_sweep_s0"
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

tag_from_beta() {
  local b="$1"
  echo "b$(echo "$b" | sed 's/-/m/g; s/\\./p/g')"
}

echo ">>> [Preflight] 检查关键输入"
for f in "${WAY_ROUTES}" "${WAY_GRAPH}" "${WAY_FEATURES}" "${WAY_REGIONS}" "${REGION_SEQ}" "${SPLIT_JSON}" "${CITY_META}"; do
  if [ -f "${f}" ]; then
    ls -lh "${f}"
  else
    echo "[MISS] ${f}"
  fi
done

BETAS=("0.001" "0.005" "0.02" "0.05")

for BETA in "${BETAS[@]}"; do
  TAG="$(tag_from_beta "${BETA}")"
  EXP_DIR="${OUT_ROOT}/${TAG}"
  OUT_A1="${EXP_DIR}/A1_beta_vae64_ae"
  OUT_A2="${EXP_DIR}/A2_flow_on_mu64"
  OUT_A3="${EXP_DIR}/A3_eval_k16_antiloop"
  OUT_A4="${EXP_DIR}/A4_phaseC_covdiv"
  mkdir -p "${OUT_A1}" "${OUT_A2}" "${OUT_A3}" "${OUT_A4}"

  echo ""
  echo "#################### beta=${BETA} (${TAG}) ####################"

  if [ ! -f "${OUT_A1}/ckpt_best.pt" ]; then
    run_step "${TAG}_A1_train_betaVAE64" "${OUT_A1}/run_train_beta_vae64.log" \
      conda run -n dpl python -u -m src.training.train_way_casd_autoencoder \
        --way_routes_npz "${WAY_ROUTES}" \
        --way_graph_npz "${WAY_GRAPH}" \
        --way_features_npz "${WAY_FEATURES}" \
        --split_json "${SPLIT_JSON}" \
        --out_dir "${OUT_A1}" \
        --batch_size 256 \
        --num_workers 24 \
        --n_epochs 100 \
        --lr 2e-4 \
        --weight_decay 1e-4 \
        --min_hops 5 \
        --max_way_len 160 \
        --max_len 160 \
        --max_candidates 32 \
        --d_model 256 \
        --n_latent 8 \
        --n_heads 8 \
        --dropout 0.1 \
        --decoder_use_cross_attn \
        --decoder_use_cand_query \
        --decoder_use_past_context \
        --decoder_past_k 16 \
        --vae_dim 64 \
        --vae_beta "${BETA}" \
        --vae_beta_warmup_epochs 30 \
        --save_every 10 \
        --early_stop_patience 20 \
        --device cuda \
        --seed 0
  else
    echo ">>> [${TAG}_A1_train_betaVAE64] SKIP (ckpt_best.pt exists)"
  fi

  if [ ! -f "${OUT_A2}/ckpt_best.pt" ]; then
    FLOW_RESUME_ARGS=()
    if [ -f "${OUT_A2}/ckpt_last.pt" ]; then
      echo ">>> [${TAG}_A2_train_flow_mu64] RESUME from ${OUT_A2}/ckpt_last.pt"
      FLOW_RESUME_ARGS+=(--resume_ckpt "${OUT_A2}/ckpt_last.pt")
    fi
    run_step "${TAG}_A2_train_flow_mu64" "${OUT_A2}/run_train_flow_mu64.log" \
      conda run -n dpl python -u -m src.training.train_way_casd_flow \
        --way_routes_npz "${WAY_ROUTES}" \
        --way_graph_npz "${WAY_GRAPH}" \
        --way_features_npz "${WAY_FEATURES}" \
        --ae_ckpt "${OUT_A1}/ckpt_best.pt" \
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
        --save_every 10 \
        --early_stop_patience 15 \
        --device cuda \
        --seed 0 \
        "${FLOW_RESUME_ARGS[@]}"
  else
    echo ">>> [${TAG}_A2_train_flow_mu64] SKIP (ckpt_best.pt exists)"
  fi

  if [ ! -f "${OUT_A2}/ckpt_best.pt" ]; then
    echo ">>> [${TAG}_A3_eval_k16_AL] SKIP (A2 ckpt_best.pt missing)"
  elif [ ! -f "${OUT_A3}/binned_betaVAE64_${TAG}_k16_dest_efficient_antiloop_n5000.json" ]; then
    run_step "${TAG}_A3_eval_k16_AL" "${OUT_A3}/run_eval_k16_AL_n5000.log" \
      conda run -n dpl python -u -m src.evaluation.way_casd_binned_eval \
        --way_routes_npz "${WAY_ROUTES}" \
        --way_graph_npz "${WAY_GRAPH}" \
        --way_features_npz "${WAY_FEATURES}" \
        --ae_ckpt "${OUT_A1}/ckpt_best.pt" \
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
        --out_json "${OUT_A3}/binned_betaVAE64_${TAG}_k16_dest_efficient_antiloop_n5000.json" \
        --out_per_route_json "${OUT_A3}/per_route_betaVAE64_${TAG}_k16_dest_efficient_antiloop_n5000.json" \
        --device cuda \
        --seed 0
  else
    echo ">>> [${TAG}_A3_eval_k16_AL] SKIP (binned json exists)"
  fi

  if [ ! -f "${OUT_A3}/per_route_betaVAE64_${TAG}_k16_dest_efficient_antiloop_n5000.json" ]; then
    echo ">>> [${TAG}_A4_phaseC_covdiv] SKIP (A3 per_route json missing)"
  elif [ ! -f "${OUT_A4}/od_coverage_diversity_betaVAE64_${TAG}_k16_AL_n5000_tau03.json" ]; then
    run_step "${TAG}_A4_phaseC_covdiv" "${OUT_A4}/run_od_coverage_diversity_k16_tau03.log" \
      conda run -n dpl python -u -m src.evaluation.od_coverage_diversity_eval \
        --method "BetaVAE64_${TAG}_K16_AL|greedy=${OUT_A3}/per_route_betaVAE64_${TAG}_k16_dest_efficient_antiloop_n5000.json" \
        --k 16 \
        --min_routes_per_od 3 \
        --jaccard_threshold 0.3 \
        --tau_values "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
        --save_per_od \
        --out_json "${OUT_A4}/od_coverage_diversity_betaVAE64_${TAG}_k16_AL_n5000_tau03.json"
  else
    echo ">>> [${TAG}_A4_phaseC_covdiv] SKIP (phaseC json exists)"
  fi
done

echo ""
echo ">>> [Summary] 汇总 beta sweep 关键指标"
python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("_sync/wsa/pi_verify/20260302_porto_beta_vae64_beta_sweep_s0")
betas = [0.001, 0.005, 0.02, 0.05]

rows = []
for b in betas:
    tag = f"b{str(b).replace('-', 'm').replace('.', 'p')}"
    p_bin = root / tag / "A3_eval_k16_antiloop" / f"binned_betaVAE64_{tag}_k16_dest_efficient_antiloop_n5000.json"
    p_cov = root / tag / "A4_phaseC_covdiv" / f"od_coverage_diversity_betaVAE64_{tag}_k16_AL_n5000_tau03.json"
    if (not p_bin.exists()) or (not p_cov.exists()):
        rows.append({"beta": float(b), "ok": False, "missing": [str(p_bin), str(p_cov)]})
        continue
    bj = json.loads(p_bin.read_text(encoding="utf-8"))
    cj = json.loads(p_cov.read_text(encoding="utf-8"))
    br = bj.get("global", {})
    ms = cj.get("methods", [])
    m0 = ms[0] if isinstance(ms, list) and len(ms) > 0 else {}
    rows.append(
        {
            "beta": float(b),
            "ok": True,
            "success": float(br.get("success_rate", np.nan)),
            "hit_wall": float(br.get("hit_wall_rate", np.nan)),
            "loop": float(br.get("loop_rate", np.nan)),
            "len_ratio_mean": float(br.get("len_ratio_mean", np.nan)),
            "coverage_tau03": float(((m0.get("gt_coverage_at_k", {}) if isinstance(m0, dict) else {}).get("mean", np.nan))),
            "diversity": float(((m0.get("self_diversity_at_k", {}) if isinstance(m0, dict) else {}).get("mean", np.nan))),
            "meanmaxj": float(((m0.get("mean_max_jaccard_at_k", {}) if isinstance(m0, dict) else {}).get("mean", np.nan))),
            "covtau_auc": float((m0.get("coverage_vs_tau_auc", np.nan) if isinstance(m0, dict) else np.nan)),
            "binned_json": str(p_bin),
            "phasec_json": str(p_cov),
        }
    )

out = {"ok": True, "task": "beta_vae64_beta_sweep_summary", "rows": rows}
out_path = root / "beta_sweep_summary.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] saved: {out_path}")
print("beta | success | hit_wall | loop | len_ratio_mean | cov@0.3 | div | meanmaxj | covtau_auc")
print("-" * 100)
for r in rows:
    if not r.get("ok", False):
        print(f"{r['beta']:.3f} | MISSING")
        continue
    print(
        f"{r['beta']:.3f} | {r['success']:.4f} | {r['hit_wall']:.4f} | {r['loop']:.4f} | "
        f"{r['len_ratio_mean']:.4f} | {r['coverage_tau03']:.4f} | {r['diversity']:.4f} | "
        f"{r['meanmaxj']:.4f} | {r['covtau_auc']:.4f}"
    )
PY

echo ""
if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
  echo ">>> DONE with failures: ${FAILED_STEPS[*]}"
else
  echo ">>> DONE: beta sweep 全流程完成"
fi

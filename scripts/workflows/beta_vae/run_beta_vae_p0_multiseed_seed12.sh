#!/usr/bin/env bash

# P0: 多种子完整流水线（seed=1,2）
# 口径固定：beta-VAE(d=64,beta=0.01,warmup=30) + Flow(mu64,n_layers=6) + K16 dest_efficient + anti-loop + PhaseC
# 说明：
# - 单卡 GPU 任务严格串行（避免显存争抢）
# - CPU 的 PhaseC 分析在对应 seed eval 后立即执行
# - 若目标产物已存在则自动跳过（防重复跑）

set -u

echo ">>> [Init] 进入仓库目录"
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

BASE_S0="_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0"
S0_BINNED="${BASE_S0}/B2_eval_k16_antiloop/binned_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json"
S0_PHASEC="_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/C1_od_coverage_b2/od_coverage_diversity_b2_k16_n5000.json"

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

check_inputs() {
  echo ">>> [Preflight] 检查关键输入"
  for f in \
    "${WAY_ROUTES}" "${WAY_GRAPH}" "${WAY_FEATURES}" "${WAY_REGIONS}" \
    "${REGION_SEQ}" "${SPLIT_JSON}" "${CITY_META}" "${S0_BINNED}" "${S0_PHASEC}"
  do
    if [ -f "${f}" ]; then
      ls -lh "${f}"
    else
      echo "[MISS] ${f}"
    fi
  done
}

run_for_seed() {
  local SEED="$1"
  local OUT_ROOT="_sync/wsa/pi_verify/20260225_porto_beta_vae64_flowmu_s${SEED}"
  local OUT_A1="${OUT_ROOT}/A1_beta_vae64_ae"
  local OUT_A2="${OUT_ROOT}/A2_flow_on_mu64"
  local OUT_A3="${OUT_ROOT}/A3_eval_k16_antiloop"
  local OUT_A4="${OUT_ROOT}/A4_phaseC_covdiv"

  mkdir -p "${OUT_A1}" "${OUT_A2}" "${OUT_A3}" "${OUT_A4}"
  echo ""
  echo "######################################################################"
  echo ">>> [Seed ${SEED}] OUT_ROOT=${OUT_ROOT}"
  echo "######################################################################"

  if [ ! -f "${OUT_A1}/ckpt_best.pt" ]; then
    run_step "S${SEED}_A1_train_betaVAE64_AE" "${OUT_A1}/run_train_beta_vae64.log" \
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
        --vae_beta 0.01 \
        --vae_beta_warmup_epochs 30 \
        --save_every 10 \
        --early_stop_patience 20 \
        --device cuda \
        --seed "${SEED}"
  else
    echo ">>> [S${SEED}_A1_train_betaVAE64_AE] SKIP (ckpt_best.pt already exists)"
  fi

  if [ ! -f "${OUT_A2}/ckpt_best.pt" ]; then
    run_step "S${SEED}_A2_train_flow_on_mu64" "${OUT_A2}/run_train_flow_mu64.log" \
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
        --n_epochs 60 \
        --lr 2e-4 \
        --weight_decay 1e-4 \
        --min_hops 5 \
        --max_way_len 160 \
        --max_candidates 32 \
        --flow_target vae_mu \
        --cond_inject xattn \
        --n_layers 6 \
        --save_every 10 \
        --early_stop_patience 15 \
        --device cuda \
        --seed "${SEED}"
  else
    echo ">>> [S${SEED}_A2_train_flow_on_mu64] SKIP (ckpt_best.pt already exists)"
  fi

  if [ ! -f "${OUT_A3}/binned_betaVAE64_flowmu_k16_dest_efficient_antiloop_n5000.json" ]; then
    run_step "S${SEED}_A3_eval_k16_dest_efficient_AL" "${OUT_A3}/run_eval_k16_dest_efficient_AL.log" \
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
        --out_json "${OUT_A3}/binned_betaVAE64_flowmu_k16_dest_efficient_antiloop_n5000.json" \
        --out_per_route_json "${OUT_A3}/per_route_betaVAE64_flowmu_k16_dest_efficient_antiloop_n5000.json" \
        --device cuda \
        --seed "${SEED}"
  else
    echo ">>> [S${SEED}_A3_eval_k16_dest_efficient_AL] SKIP (binned exists)"
  fi

  if [ ! -f "${OUT_A4}/od_coverage_diversity_betaVAE64_k16_AL_n5000_tau03.json" ]; then
    run_step "S${SEED}_A4_phaseC_covdiv_k16" "${OUT_A4}/run_od_coverage_diversity_k16_tau03.log" \
      conda run -n dpl python -u -m src.evaluation.od_coverage_diversity_eval \
        --method "BetaVAE64_FlowMu_K16_AL_s${SEED}|greedy=${OUT_A3}/per_route_betaVAE64_flowmu_k16_dest_efficient_antiloop_n5000.json" \
        --k 16 \
        --min_routes_per_od 3 \
        --jaccard_threshold 0.3 \
        --tau_values "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
        --save_per_od \
        --out_json "${OUT_A4}/od_coverage_diversity_betaVAE64_k16_AL_n5000_tau03.json"
  else
    echo ">>> [S${SEED}_A4_phaseC_covdiv_k16] SKIP (phaseC json exists)"
  fi
}

echo ">>> [Step 0] 预检"
check_inputs

run_for_seed 1
run_for_seed 2

echo ""
echo ">>> [Summary] 聚合 seed=0/1/2（mean±std）"
python - <<'PY'
import json
import statistics as st
from pathlib import Path

def load_weighted_binned(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    cells = (((d.get("overall") or {}).get("greedy") or {}).get("cells") or {})
    if not isinstance(cells, dict) or not cells:
        return None
    n = sum(float(v.get("n", 0) or 0) for v in cells.values())
    ns = sum(float(v.get("n_success", 0) or 0) for v in cells.values())
    if n <= 0:
        return None
    def wavg(k):
        return sum(float(v.get("n", 0) or 0) * float(v.get(k, 0) or 0) for v in cells.values()) / n
    out = {
        "success": wavg("success_rate"),
        "hit_wall": wavg("hit_wall_rate"),
        "loop": wavg("loop_rate"),
        "len_ratio_mean": sum(float(v.get("n", 0) or 0) * float(((v.get("len_ratio") or {}).get("mean", 0) or 0)) for v in cells.values()) / n,
    }
    if ns > 0:
        out["succ_len_ratio_p50_weighted"] = (
            sum(float(v.get("n_success", 0) or 0) * float((((v.get("success_only_len_ratio") or {}).get("p50")) or 0)) for v in cells.values()) / ns
        )
    else:
        out["succ_len_ratio_p50_weighted"] = None
    return out

def load_phasec(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    table = d.get("summary_table") or []
    if not table:
        return None
    row = table[0]
    return {
        "arrival_phasec": float(row.get("arrival_rate", 0.0)),
        "coverage": float(row.get("gt_coverage_at_k_mean", 0.0)),
        "diversity": float(row.get("self_diversity_at_k_mean", 0.0)),
        "meanmaxj": float(row.get("mean_max_jaccard_at_k_mean", 0.0)),
        "covtau_auc": float(row.get("coverage_vs_tau_auc", 0.0)),
    }

items = {
    0: {
        "binned": Path("_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B2_eval_k16_antiloop/binned_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json"),
        "phasec": Path("_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/C1_od_coverage_b2/od_coverage_diversity_b2_k16_n5000.json"),
    },
    1: {
        "binned": Path("_sync/wsa/pi_verify/20260225_porto_beta_vae64_flowmu_s1/A3_eval_k16_antiloop/binned_betaVAE64_flowmu_k16_dest_efficient_antiloop_n5000.json"),
        "phasec": Path("_sync/wsa/pi_verify/20260225_porto_beta_vae64_flowmu_s1/A4_phaseC_covdiv/od_coverage_diversity_betaVAE64_k16_AL_n5000_tau03.json"),
    },
    2: {
        "binned": Path("_sync/wsa/pi_verify/20260225_porto_beta_vae64_flowmu_s2/A3_eval_k16_antiloop/binned_betaVAE64_flowmu_k16_dest_efficient_antiloop_n5000.json"),
        "phasec": Path("_sync/wsa/pi_verify/20260225_porto_beta_vae64_flowmu_s2/A4_phaseC_covdiv/od_coverage_diversity_betaVAE64_k16_AL_n5000_tau03.json"),
    },
}

rows = {}
for s, p in items.items():
    b = load_weighted_binned(p["binned"])
    c = load_phasec(p["phasec"])
    if b is None or c is None:
        rows[s] = None
    else:
        rows[s] = {"seed": s, **b, **c}

ok = [r for r in rows.values() if isinstance(r, dict)]
metrics = ["success","hit_wall","loop","len_ratio_mean","arrival_phasec","coverage","diversity","meanmaxj","covtau_auc"]
agg = {}
for m in metrics:
    vals = [float(r[m]) for r in ok if r.get(m) is not None]
    if vals:
        agg[m] = {"mean": float(st.mean(vals)), "std": float(st.pstdev(vals)), "n": len(vals)}

out = {
    "ok": True,
    "task": "beta_vae64_multiseed_summary",
    "seeds": rows,
    "aggregate_mean_std": agg,
}
out_path = Path("_sync/wsa/pi_verify/20260225_porto_beta_vae64_multiseed_summary.json")
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] saved: {out_path}")
print("metric | mean | std | n")
print("-" * 44)
for m in metrics:
    if m in agg:
        a = agg[m]
        print(f"{m:16s} | {a['mean']:.4f} | {a['std']:.4f} | {a['n']}")
PY

echo ""
echo "======================================================================"
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
  echo "DONE with failures: ${FAILED_STEPS[*]}"
else
  echo "DONE: P0 多种子流水线完成（seed=1,2 + mean±std聚合）"
fi
echo "======================================================================"


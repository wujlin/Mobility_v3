#!/usr/bin/env bash

# Beta-VAE vae_dim=32 ablation (Porto, seed0)
# Pipeline:
#   A1) train beta-VAE AE (vae_dim=32)
#   A2) train Flow on mu32 (flow_target=vae_mu)
#   A3) eval K16 dest_efficient (no anti-loop)
#   A4) eval K16 dest_efficient (anti-loop k=4, penalty=2.0)
#   A5) Phase-C coverage/diversity from A4 per_route
#
# Notes:
# - No set -e: keep running and collect failures per step.
# - Real-time logs via tee.

set -u

echo ">>> [Init] 进入仓库目录"
cd ~/projects/Mobility_v3 || cd /home/jinlin/projects/Mobility_v3 || true

OUT_ROOT="_sync/wsa/pi_verify/20260224_porto_beta_vae32_flowmu_s0"
OUT_A1="${OUT_ROOT}/A1_beta_vae32_ae"
OUT_A2="${OUT_ROOT}/A2_flow_on_mu32"
OUT_A3="${OUT_ROOT}/A3_eval_k16_noAL"
OUT_A4="${OUT_ROOT}/A4_eval_k16_AL"
OUT_A5="${OUT_ROOT}/A5_phaseC_covdiv"

DATA_BASE="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${DATA_BASE}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${DATA_BASE}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_BASE}/W3_way_features/way_features.npz"
WAY_REGIONS="${DATA_BASE}/region_sweep/way_regions_louvain_res5_seed0.npz"
REGION_SEQ="${DATA_BASE}/region_seq_res5/region_seq_min3_max160.npz"
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

echo ">>> [Preflight] 创建输出目录"
mkdir -p "${OUT_A1}" "${OUT_A2}" "${OUT_A3}" "${OUT_A4}" "${OUT_A5}"

echo ">>> [Preflight] 检查关键输入"
for f in \
  "${WAY_ROUTES}" "${WAY_GRAPH}" "${WAY_FEATURES}" "${WAY_REGIONS}" "${REGION_SEQ}" "${SPLIT_JSON}" "${CITY_META}"
do
  if [ -f "${f}" ]; then
    ls -lh "${f}"
  else
    echo "[MISS] ${f}"
  fi
done

echo ""
echo ">>> [A1] 训练 beta-VAE AE (vae_dim=32)"
run_step "A1_train_betaVAE32_AE" "${OUT_A1}/run_train_beta_vae32.log" \
  conda run -n dpl python -u -m src.training.train_way_casd_autoencoder \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --split_json "${SPLIT_JSON}" \
    --out_dir "${OUT_A1}" \
    --batch_size 256 \
    --num_workers 24 \
    --n_epochs 120 \
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
    --vae_dim 32 \
    --vae_beta 0.01 \
    --vae_beta_warmup_epochs 30 \
    --save_every 10 \
    --early_stop_patience 20 \
    --device cuda \
    --seed 0

echo ""
echo ">>> [A2] 训练 Flow on mu32"
run_step "A2_train_flow_on_mu32" "${OUT_A2}/run_train_flow_mu32.log" \
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
    --flow_target vae_mu \
    --cond_inject xattn \
    --n_layers 6 \
    --save_every 10 \
    --early_stop_patience 15 \
    --device cuda \
    --seed 0

echo ""
echo ">>> [A3] 评估 K16 dest_efficient (no anti-loop)"
run_step "A3_eval_k16_dest_efficient_noAL" "${OUT_A3}/run_eval_k16_dest_efficient_noAL.log" \
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
    --anti_loop_k 0 \
    --anti_loop_penalty 0.0 \
    --anti_loop_penalty_k 4 \
    --no_compare_beam \
    --city_grid_meta "0=${CITY_META}" \
    --eval_batch_size 256 \
    --dump_way_seqs \
    --out_json "${OUT_A3}/binned_betaVAE32_flowmu_k16_dest_efficient_n5000.json" \
    --out_per_route_json "${OUT_A3}/per_route_betaVAE32_flowmu_k16_dest_efficient_n5000.json" \
    --device cuda \
    --seed 0

echo ""
echo ">>> [A4] 评估 K16 dest_efficient + anti-loop"
run_step "A4_eval_k16_dest_efficient_AL" "${OUT_A4}/run_eval_k16_dest_efficient_AL.log" \
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
    --out_json "${OUT_A4}/binned_betaVAE32_flowmu_k16_dest_efficient_antiloop_n5000.json" \
    --out_per_route_json "${OUT_A4}/per_route_betaVAE32_flowmu_k16_dest_efficient_antiloop_n5000.json" \
    --device cuda \
    --seed 0

echo ""
echo ">>> [A5] Phase-C (A4 per_route)"
run_step "A5_phaseC_covdiv_k16" "${OUT_A5}/run_od_coverage_diversity_k16.log" \
  conda run -n dpl python -u -m src.evaluation.od_coverage_diversity_eval \
    --method "BetaVAE32_FlowMu_K16_AL|greedy=${OUT_A4}/per_route_betaVAE32_flowmu_k16_dest_efficient_antiloop_n5000.json" \
    --k 16 \
    --min_routes_per_od 3 \
    --jaccard_threshold 0.3 \
    --save_per_od \
    --out_json "${OUT_A5}/od_coverage_diversity_betaVAE32_flowmu_k16_AL_n5000.json"

echo ""
echo ">>> [Summary] 提取关键指标"
python - <<'PY'
import json
from pathlib import Path

targets = {
    "A3_noAL": Path("_sync/wsa/pi_verify/20260224_porto_beta_vae32_flowmu_s0/A3_eval_k16_noAL/binned_betaVAE32_flowmu_k16_dest_efficient_n5000.json"),
    "A4_AL": Path("_sync/wsa/pi_verify/20260224_porto_beta_vae32_flowmu_s0/A4_eval_k16_AL/binned_betaVAE32_flowmu_k16_dest_efficient_antiloop_n5000.json"),
}

def weighted_from_binned(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    cells = (((d.get("overall") or {}).get("greedy") or {}).get("cells") or {})
    if not isinstance(cells, dict) or not cells:
        return None
    n = sum(float(v.get("n", 0) or 0) for v in cells.values())
    if n <= 0:
        return None
    def wavg(key):
        return sum(float(v.get("n", 0) or 0) * float(v.get(key, 0) or 0) for v in cells.values()) / n
    return {
        "success": wavg("success_rate"),
        "hit_wall": wavg("hit_wall_rate"),
        "loop": wavg("loop_rate"),
        "len_ratio_mean": sum(float(v.get("n", 0) or 0) * float(((v.get("len_ratio") or {}).get("mean", 0) or 0)) for v in cells.values()) / n,
        "success_only_len_ratio_p50": sum(float(v.get("n_success", 0) or 0) * float((((v.get("success_only_len_ratio") or {}).get("p50")) or 0)) for v in cells.values()) / max(1.0, sum(float(v.get("n_success", 0) or 0) for v in cells.values())),
    }

print("name | success | hit_wall | loop | len_ratio_mean | succ_len_ratio_p50")
print("-" * 78)
for name, p in targets.items():
    m = weighted_from_binned(p)
    if m is None:
        print(f"{name} | MISSING")
    else:
        print(f"{name} | {m['success']:.4f} | {m['hit_wall']:.4f} | {m['loop']:.4f} | {m['len_ratio_mean']:.4f} | {m['success_only_len_ratio_p50']:.4f}")
PY

echo ""
echo "======================================================================"
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
  echo "DONE with failures: ${FAILED_STEPS[*]}"
else
  echo "DONE: vae_dim=32 ablation 全部完成"
fi
echo "OUT_ROOT: ${OUT_ROOT}"
echo "======================================================================"

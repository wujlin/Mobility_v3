#!/usr/bin/env bash

# C1/C2/C3 实验流水线（WSA）
# C1: B2 的 OD coverage/diversity（零训练）
# C2: beta-VAE GT μ oracle（latent_source=gt, K=1）
# C3: 修复 vae128 Flow（n_layers=6）+ K16 eval

set -u

echo ">>> [Init] 进入仓库目录"
cd ~/projects/Mobility_v3 || cd /home/jinlin/projects/Mobility_v3 || true

BASE64="_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0"
BASE128="_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0"

OUT_C1="${BASE128}/C1_od_coverage_b2"
OUT_C2="${BASE128}/C2_gt_mu_oracle_k1"
OUT_C3_FLOW="${BASE128}/C3_flow_on_mu128_l6_fix"
OUT_C3_EVAL="${BASE128}/C3_eval_k16_l6_fix"

DATA_BASE="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${DATA_BASE}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${DATA_BASE}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_BASE}/W3_way_features/way_features.npz"
WAY_REGIONS="${DATA_BASE}/region_sweep/way_regions_louvain_res5_seed0.npz"
REGION_SEQ="${DATA_BASE}/region_seq_res5/region_seq_min3_max160.npz"
SPLIT_JSON="${DATA_BASE}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"
CITY_META="/home/jinlin/data/geoexplicit_data/porto_taxi/semantic/osm_road_prob_meta.json"

AE64="${BASE64}/A1_beta_vae_ae/ckpt_best.pt"
FLOW64="${BASE64}/A2_flow_on_mu/ckpt_best.pt"
AE128="${BASE128}/A1_beta_vae128_ae/ckpt_best.pt"

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

echo ">>> [Preflight] 创建目录"
mkdir -p "${OUT_C1}" "${OUT_C2}" "${OUT_C3_FLOW}" "${OUT_C3_EVAL}"

echo ">>> [Preflight] 检查文件"
for f in \
  "${BASE64}/B2_eval_k16_antiloop/per_route_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json" \
  "${WAY_ROUTES}" "${WAY_GRAPH}" "${WAY_FEATURES}" "${WAY_REGIONS}" "${REGION_SEQ}" "${SPLIT_JSON}" "${CITY_META}" \
  "${AE64}" "${FLOW64}" "${AE128}"
do
  if [ -f "${f}" ]; then ls -lh "${f}"; else echo "[MISS] ${f}"; fi
done

echo ""
echo ">>> [C1] B2 的 OD coverage/diversity"
run_step "C1_od_coverage_b2_k16" "${OUT_C1}/run_od_coverage_diversity_b2_k16_n5000.log" \
  conda run -n dpl python -u -m src.evaluation.od_coverage_diversity_eval \
    --method "BetaVAE64_FlowMu_K16_AL|greedy=${BASE64}/B2_eval_k16_antiloop/per_route_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json" \
    --k 16 \
    --min_routes_per_od 3 \
    --jaccard_threshold 0.5 \
    --out_json "${OUT_C1}/od_coverage_diversity_b2_k16_n5000.json"

echo ""
echo ">>> [C2] beta-VAE GT μ oracle（K=1）"
run_step "C2_gt_mu_oracle_k1" "${OUT_C2}/run_eval_betaVAE64_gt_k1_n5000.log" \
  conda run -n dpl python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE64}" \
    --latent_source gt \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --n_routes 5000 \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --n_samples_per_route 1 \
    --sample_select first \
    --decode_max_candidates 0 \
    --decode_candidate_policy first \
    --anti_loop_k 0 \
    --anti_loop_penalty 0.0 \
    --anti_loop_penalty_k 4 \
    --no_compare_beam \
    --city_grid_meta "0=${CITY_META}" \
    --eval_batch_size 256 \
    --dump_way_seqs \
    --out_json "${OUT_C2}/binned_betaVAE64_gt_k1_n5000.json" \
    --out_per_route_json "${OUT_C2}/per_route_betaVAE64_gt_k1_n5000.json" \
    --device cuda \
    --seed 0

echo ""
echo ">>> [C3-Flow] 修复 vae128 Flow（n_layers=6）"
run_step "C3_train_flow_mu128_l6" "${OUT_C3_FLOW}/run_train_flow_mu128_l6.log" \
  conda run -n dpl python -u -m src.training.train_way_casd_flow \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE128}" \
    --region_seq_npz "${REGION_SEQ}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --use_region_seq \
    --split_json "${SPLIT_JSON}" \
    --out_dir "${OUT_C3_FLOW}" \
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
echo ">>> [C3-Eval] vae128-l6 Flow，K=16 dest_efficient"
run_step "C3_eval_k16_dest_efficient" "${OUT_C3_EVAL}/run_eval_k16_dest_efficient.log" \
  conda run -n dpl python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE128}" \
    --flow_ckpt "${OUT_C3_FLOW}/ckpt_best.pt" \
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
    --out_json "${OUT_C3_EVAL}/binned_betaVAE128_flowmu_l6_k16_dest_efficient_n5000.json" \
    --out_per_route_json "${OUT_C3_EVAL}/per_route_betaVAE128_flowmu_l6_k16_dest_efficient_n5000.json" \
    --device cuda \
    --seed 0

echo ""
echo ">>> [Summary] 关键指标提取"
python - <<'PY'
import json
from pathlib import Path

def from_binned(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    cells = d.get("overall", {}).get("greedy", {}).get("cells", {})
    if not cells:
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
        "len_ratio_mean": sum(float(v.get("n", 0) or 0) * float(v.get("len_ratio", {}).get("mean", 0) or 0) for v in cells.values()) / n,
    }

targets = {
    "C2_gt_mu_k1": Path("_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/C2_gt_mu_oracle_k1/binned_betaVAE64_gt_k1_n5000.json"),
    "C3_mu128_l6_k16": Path("_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/C3_eval_k16_l6_fix/binned_betaVAE128_flowmu_l6_k16_dest_efficient_n5000.json"),
}
print("name | success | hit_wall | loop | len_ratio_mean")
print("-" * 66)
for name, p in targets.items():
    m = from_binned(p)
    if m is None:
        print(f"{name} | MISSING")
    else:
        print(f"{name} | {m['success']:.4f} | {m['hit_wall']:.4f} | {m['loop']:.4f} | {m['len_ratio_mean']:.4f}")
PY

echo ""
echo "======================================================================"
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
  echo "DONE with failures: ${FAILED_STEPS[*]}"
else
  echo "DONE: C1/C2/C3 全部完成"
fi
echo "======================================================================"


#!/usr/bin/env bash

# B1/B2/B3 一键实验流水线（WSA）
# - B1: 现有 beta-VAE + Flow(mu) 跑 K=16 dest_efficient
# - B2: 在 B1 基础上开启 anti-loop
# - B3: 训练高容量 beta-VAE(vae_dim=128) + Flow(mu) + K=16 eval
# 说明：单卡 GPU 下重任务串行，避免显存争抢导致整体变慢。

set -u

echo ">>> [Init] 进入仓库根目录"
cd ~/projects/Mobility_v3 || cd /home/jinlin/projects/Mobility_v3 || true

OUT_BASE="_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0"
OUT_B3="_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0"

OUT_B1="${OUT_BASE}/B1_eval_k16"
OUT_B2="${OUT_BASE}/B2_eval_k16_antiloop"
OUT_B3_AE="${OUT_B3}/A1_beta_vae128_ae"
OUT_B3_FLOW="${OUT_B3}/A2_flow_on_mu128"
OUT_B3_EVAL="${OUT_B3}/A3_eval_k16"

DATA_BASE="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${DATA_BASE}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${DATA_BASE}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_BASE}/W3_way_features/way_features.npz"
WAY_REGIONS="${DATA_BASE}/region_sweep/way_regions_louvain_res5_seed0.npz"
REGION_SEQ="${DATA_BASE}/region_seq_res5/region_seq_min3_max160.npz"
SPLIT_JSON="${DATA_BASE}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"
CITY_META="/home/jinlin/data/geoexplicit_data/porto_taxi/semantic/osm_road_prob_meta.json"

AE_A1="${OUT_BASE}/A1_beta_vae_ae/ckpt_best.pt"
FLOW_A2="${OUT_BASE}/A2_flow_on_mu/ckpt_best.pt"

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
mkdir -p "${OUT_B1}" "${OUT_B2}" "${OUT_B3_AE}" "${OUT_B3_FLOW}" "${OUT_B3_EVAL}"

echo ">>> [Preflight] 检查关键输入"
for f in "${WAY_ROUTES}" "${WAY_GRAPH}" "${WAY_FEATURES}" "${WAY_REGIONS}" "${REGION_SEQ}" "${SPLIT_JSON}" "${CITY_META}" "${AE_A1}" "${FLOW_A2}"; do
  if [ -f "${f}" ]; then
    ls -lh "${f}"
  else
    echo "[MISS] ${f}"
  fi
done

echo ""
echo ">>> [B1] 现有 A1+A2：K=16 dest_efficient（无 anti-loop）"
run_step "B1_eval_k16_dest_efficient" "${OUT_B1}/run_eval_k16_dest_efficient.log" \
  conda run -n dpl python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_A1}" \
    --flow_ckpt "${FLOW_A2}" \
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
    --out_json "${OUT_B1}/binned_betaVAE_flowmu_k16_dest_efficient_n5000.json" \
    --out_per_route_json "${OUT_B1}/per_route_betaVAE_flowmu_k16_dest_efficient_n5000.json" \
    --device cuda \
    --seed 0

echo ""
echo ">>> [B2] 现有 A1+A2：K=16 dest_efficient + anti-loop"
run_step "B2_eval_k16_dest_efficient_antiloop" "${OUT_B2}/run_eval_k16_dest_efficient_antiloop.log" \
  conda run -n dpl python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_A1}" \
    --flow_ckpt "${FLOW_A2}" \
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
    --out_json "${OUT_B2}/binned_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json" \
    --out_per_route_json "${OUT_B2}/per_route_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json" \
    --device cuda \
    --seed 0

echo ""
echo ">>> [B3-AE] 高容量 beta-VAE（vae_dim=128, beta=0.005, warmup=30, 150ep）"
run_step "B3_train_betaVAE128_AE" "${OUT_B3_AE}/run_train_beta_vae128.log" \
  conda run -n dpl python -u -m src.training.train_way_casd_autoencoder \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --split_json "${SPLIT_JSON}" \
    --out_dir "${OUT_B3_AE}" \
    --batch_size 256 \
    --num_workers 24 \
    --n_epochs 150 \
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
    --vae_dim 128 \
    --vae_beta 0.005 \
    --vae_beta_warmup_epochs 30 \
    --save_every 10 \
    --early_stop_patience 20 \
    --device cuda \
    --seed 0

echo ""
echo ">>> [B3-Flow] 在 μ∈R^128 上训练 Flow"
run_step "B3_train_flow_on_mu128" "${OUT_B3_FLOW}/run_train_flow_mu128.log" \
  conda run -n dpl python -u -m src.training.train_way_casd_flow \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${OUT_B3_AE}/ckpt_best.pt" \
    --region_seq_npz "${REGION_SEQ}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --use_region_seq \
    --split_json "${SPLIT_JSON}" \
    --out_dir "${OUT_B3_FLOW}" \
    --batch_size 512 \
    --num_workers 24 \
    --n_epochs 80 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --min_hops 5 \
    --max_way_len 160 \
    --max_candidates 32 \
    --n_layers 4 \
    --flow_target vae_mu \
    --cond_inject xattn \
    --save_every 10 \
    --early_stop_patience 15 \
    --device cuda \
    --seed 0

echo ""
echo ">>> [B3-Eval] 高容量 beta-VAE + Flow(mu128) K=16 dest_efficient"
run_step "B3_eval_k16_dest_efficient" "${OUT_B3_EVAL}/run_eval_k16_dest_efficient.log" \
  conda run -n dpl python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${OUT_B3_AE}/ckpt_best.pt" \
    --flow_ckpt "${OUT_B3_FLOW}/ckpt_best.pt" \
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
    --out_json "${OUT_B3_EVAL}/binned_betaVAE128_flowmu_k16_dest_efficient_n5000.json" \
    --out_per_route_json "${OUT_B3_EVAL}/per_route_betaVAE128_flowmu_k16_dest_efficient_n5000.json" \
    --device cuda \
    --seed 0

echo ""
echo ">>> [Summary] 提取关键指标"
python - <<'PY'
import json
from pathlib import Path

targets = {
    "B1": Path("_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B1_eval_k16/binned_betaVAE_flowmu_k16_dest_efficient_n5000.json"),
    "B2": Path("_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B2_eval_k16_antiloop/binned_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json"),
    "B3": Path("_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/A3_eval_k16/binned_betaVAE128_flowmu_k16_dest_efficient_n5000.json"),
}
print("name | success | hit_wall | loop | len_ratio_mean | succ_len_ratio_p50")
print("-" * 78)
for name, p in targets.items():
    if not p.exists():
        print(f"{name} | MISSING")
        continue
    d = json.loads(p.read_text())
    g = d.get("global", {})
    succ = float(g.get("success_rate", 0.0))
    hw = float(g.get("hit_wall_rate", 0.0))
    lp = float(g.get("loop_rate", 0.0))
    lr = float(g.get("len_ratio_mean", 0.0))
    slr = g.get("success_only_len_ratio_p50", None)
    slr = float(slr) if slr is not None else None
    print(f"{name} | {succ:.4f} | {hw:.4f} | {lp:.4f} | {lr:.4f} | {slr if slr is not None else 'NA'}")
PY

echo ""
echo "======================================================================"
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
  echo "DONE with failures: ${FAILED_STEPS[*]}"
else
  echo "DONE: B1/B2/B3 全部完成"
fi
echo "======================================================================"


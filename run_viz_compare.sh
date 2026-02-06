#!/bin/bash
# 快速生成可视化对比数据（仅少量样本，带 --dump_way_seqs）
# 用于验证各方法的轨迹质量

set -e

# ============ 数据路径（需要根据实际环境调整） ============
export WAY_ROUTES_NPZ="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/way_routes_strict_v1.npz"
export WAY_GRAPH_NPZ="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/way_graph.npz"
export WAY_FEATS_NPZ="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/way_features.npz"
export DET_META="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/detroit/grid_meta.json"
export COL_META="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/columbus/grid_meta.json"

# Baseline checkpoints
export RNN_CKPT="_sync/wsa/baselines_sota_s0/B2_rnn_ar_seed0/ckpt_best.pt"
export TR_CKPT="_sync/wsa/baselines_sota_s0/B3_transformer_ar_seed0/ckpt_best.pt"
export GTG_CKPT="_sync/wsa/baselines_sota_s0/S1_gtg_seed0/ckpt_best.pt"
export DIFFTRAJ_CKPT="_sync/wsa/baselines_sota_s0/S2_difftraj_seed0/ckpt_best.pt"

# Way-CASD checkpoints
export AE_CKPT="_sync/wsa/W9_ae_regcond_past16candq_s0/checkpoints/best_ckpt.pt"
export FLOW_CKPT="_sync/wsa/W11_flow_regseq_xattn_s0/ckpt_best.pt"
export WAY_REGIONS_NPZ="_sync/wsa/pi_verify/20260201_min5_candq1_past8_len160_s0/regions_louvain_res1p0_v1/way_regions.npz"
export REGION_AR_CKPT="_sync/wsa/W10_region_ar_l1p0_s0/ckpt_best.pt"

# 输出目录
export OUT_DIR="_sync/wsa/viz_compare_s0"
mkdir -p "$OUT_DIR"

N_ROUTES=50  # 少量样本，快速验证

echo "============ 1. Shortest Path ============"
PYTHONUNBUFFERED=1 python -u -m src.evaluation.unified_binned_eval \
  --method shortest_path \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" --city_grid_meta "1=$COL_META" \
  --n_routes $N_ROUTES --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --beam_size 10 --seed 0 \
  --dump_way_seqs \
  --out_json "$OUT_DIR/sp_n${N_ROUTES}.json" \
  |& tee "$OUT_DIR/run_sp.log"

echo "============ 2. RNN-AR ============"
PYTHONUNBUFFERED=1 python -u -m src.evaluation.unified_binned_eval \
  --method rnn_ar \
  --ckpt "$RNN_CKPT" \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" --city_grid_meta "1=$COL_META" \
  --n_routes $N_ROUTES --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --beam_size 10 --seed 0 \
  --dump_way_seqs \
  --out_json "$OUT_DIR/rnn_ar_n${N_ROUTES}.json" \
  |& tee "$OUT_DIR/run_rnn_ar.log"

echo "============ 3. Transformer-AR ============"
PYTHONUNBUFFERED=1 python -u -m src.evaluation.unified_binned_eval \
  --method transformer_ar \
  --ckpt "$TR_CKPT" \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" --city_grid_meta "1=$COL_META" \
  --n_routes $N_ROUTES --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --beam_size 10 --seed 0 \
  --dump_way_seqs \
  --out_json "$OUT_DIR/tr_ar_n${N_ROUTES}.json" \
  |& tee "$OUT_DIR/run_tr_ar.log"

echo "============ 4. GTG ============"
PYTHONUNBUFFERED=1 python -u -m src.evaluation.unified_binned_eval \
  --method gtg \
  --ckpt "$GTG_CKPT" \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" --city_grid_meta "1=$COL_META" \
  --n_routes $N_ROUTES --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --beam_size 10 --seed 0 \
  --decode_max_candidates 0 \
  --dump_way_seqs \
  --out_json "$OUT_DIR/gtg_n${N_ROUTES}.json" \
  |& tee "$OUT_DIR/run_gtg.log"

echo "============ 5. DiffTraj ============"
PYTHONUNBUFFERED=1 python -u -m src.evaluation.unified_binned_eval \
  --method difftraj \
  --ckpt "$DIFFTRAJ_CKPT" \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" --city_grid_meta "1=$COL_META" \
  --n_routes $N_ROUTES --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --beam_size 10 --seed 0 \
  --dump_way_seqs \
  --out_json "$OUT_DIR/difftraj_n${N_ROUTES}.json" \
  |& tee "$OUT_DIR/run_difftraj.log"

echo "============ 6. Way-CASD (E2 Joint FT) ============"
PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" --city_grid_meta "1=$COL_META" \
  --ae_ckpt "$AE_CKPT" \
  --latent_source flow --flow_ckpt "$FLOW_CKPT" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --region_constraint ar --region_ar_ckpt "$REGION_AR_CKPT" \
  --region_constraint_mode relaxed --region_constraint_fallback dest_region \
  --n_routes $N_ROUTES --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --decode_candidate_policy first --decode_max_candidates 0 \
  --beam_size 10 \
  --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
  --dump_way_seqs \
  --out_json "$OUT_DIR/waycasd_e2_n${N_ROUTES}.json" \
  |& tee "$OUT_DIR/run_waycasd.log"

echo ""
echo "============ 生成可视化 ============"
python -m tools.viz_method_comparison \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --result_jsons \
    "SP=$OUT_DIR/sp_n${N_ROUTES}.json" \
    "RNN-AR=$OUT_DIR/rnn_ar_n${N_ROUTES}.json" \
    "Tr-AR=$OUT_DIR/tr_ar_n${N_ROUTES}.json" \
    "GTG=$OUT_DIR/gtg_n${N_ROUTES}.json" \
    "DiffTraj=$OUT_DIR/difftraj_n${N_ROUTES}.json" \
    "WayCasd=$OUT_DIR/waycasd_e2_n${N_ROUTES}.json" \
  --out_dir "$OUT_DIR/figures" \
  --n_samples 20 \
  --seed 42 \
  --city 0

echo ""
echo "[DONE] Visualizations saved to $OUT_DIR/figures/"

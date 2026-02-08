#!/usr/bin/env bash
# ============================================================
# Porto 数据诊断：一站式运行质量审计 + 图审计 + 最短路径分析
# 目的：回答 Porto 是否解决了 Detroit 的 "GT ≈ SP" 问题
# ============================================================
set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-$HOME/projects/Mobility_v3}"
RAW_ROOT="${RAW_ROOT:-$HOME/data/geoexplicit_data}"
OUT_BASE="${RAW_ROOT}/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"

WAY_ROUTES="${OUT_BASE}/W4_way_routes_labeled/way_routes_labeled.npz"
WAY_GRAPH="${OUT_BASE}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${OUT_BASE}/W3_way_features/way_features.npz"
CITY_META="${RAW_ROOT}/porto_taxi/semantic/osm_road_prob_meta.json"

DIAG_DIR="${OUT_BASE}/A_porto_diagnose"
mkdir -p "${DIAG_DIR}"

echo "======================================"
echo "Porto 数据诊断"
echo "======================================"
echo "WAY_ROUTES: ${WAY_ROUTES}"
echo "WAY_GRAPH:  ${WAY_GRAPH}"
echo "WAY_FEATURES: ${WAY_FEATURES}"
echo "CITY_META:  ${CITY_META}"
echo "输出目录:   ${DIAG_DIR}"
echo ""

# ------------------------------------------------------------------
# Step 1: Way graph 审计 (出度分布, 连通分量)
# ------------------------------------------------------------------
echo "[Step 1/3] Way graph 审计..."
python -m src.data.way_graph.audit_way_graph_npz \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_routes_npz "${WAY_ROUTES}" \
    | tee "${DIAG_DIR}/way_graph_audit.txt"
echo ""

# ------------------------------------------------------------------
# Step 2: Way routes 质量审计 (step距离, loop, missing, filter impact)
# ------------------------------------------------------------------
echo "[Step 2/3] Way routes 质量审计..."
python -m src.data.way_graph.audit_way_routes_quality \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_features_npz "${WAY_FEATURES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --out_json "${DIAG_DIR}/way_routes_quality.json" \
    --out_bad_json "${DIAG_DIR}/way_routes_bad.json" \
    --city_meta_json "${CITY_META}" \
    --min_way_len 3 \
    --max_way_len 160
echo "  -> ${DIAG_DIR}/way_routes_quality.json"
echo ""

# ------------------------------------------------------------------
# Step 3: 最短路径基线 (detour ratio = GT长度 / SP长度)
# 这是最关键的指标：如果 detour_gt_over_sp 中位数接近 1.0，
# 说明 Porto 的 GT 就是最短路径，路径生成任务被 trivialize
# ------------------------------------------------------------------
echo "[Step 3/3] 最短路径基线 (detour ratio)..."
python -m src.evaluation.shortest_path_baseline \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --out_json "${DIAG_DIR}/shortest_path_baseline.json" \
    --city_grid_meta "0=${CITY_META}" \
    --n_routes 500 \
    --min_hops 5 \
    --max_way_len 160 \
    --seed 0
echo "  -> ${DIAG_DIR}/shortest_path_baseline.json"
echo ""

echo "======================================"
echo "诊断完成! 产物在: ${DIAG_DIR}/"
echo "======================================"
echo ""
echo "关键关注指标:"
echo "  1. way_routes_quality.json → loop_ratio, max_step_m 分布, filter impact"
echo "  2. shortest_path_baseline.json → detour_gt_over_sp (p50 >> 1.0 说明有绕行)"
echo "  3. way_graph_audit.txt → 出度分布, 连通性"
echo ""
echo "快速查看 detour ratio:"
echo "  python -c \"import json; d=json.load(open('${DIAG_DIR}/shortest_path_baseline.json')); print('detour_gt_over_sp:', d['overall']['detour_gt_over_sp'])\""

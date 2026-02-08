#!/usr/bin/env bash
# ============================================================
# Porto Strict Gate + OD-disjoint Split
#
# 目的：
#   1) 根据 A_porto_diagnose/way_routes_bad.json（默认阈值）过滤掉异常 routes
#   2) 在 strict 数据集上生成 OD-disjoint split（供训练/评测复用）
#
# 说明：
#   - 小文件（report/split json）建议同步到 repo；npz 大文件不要进 git。
# ============================================================
set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-$HOME/projects/Mobility_v3}"
RAW_ROOT="${RAW_ROOT:-$HOME/data/geoexplicit_data}"
OUT_BASE="${RAW_ROOT}/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"

WAY_ROUTES_IN="${OUT_BASE}/W4_way_routes_labeled/way_routes_labeled.npz"
BAD_JSON="${OUT_BASE}/A_porto_diagnose/way_routes_bad.json"

OUT_DIR="${OUT_BASE}/W5_way_routes_strict_gate"
WAY_ROUTES_OUT="${OUT_DIR}/way_routes_strict_gate.npz"
REPORT_JSON="${OUT_DIR}/report.json"

SPLIT_JSON="${OUT_DIR}/od_split_min3_max160_seed0.json"

mkdir -p "${OUT_DIR}"

echo "======================================"
echo "Porto Strict Gate + OD-disjoint Split"
echo "======================================"
echo "[in]  WAY_ROUTES: ${WAY_ROUTES_IN}"
echo "[in]  BAD_JSON:   ${BAD_JSON}"
echo "[out] OUT_DIR:    ${OUT_DIR}"
echo ""

echo ">>> [Step 1/2] Filtering way_routes by bad ids (default gate)..."
python -m src.data.way_graph.filter_way_routes_by_bad_ids \
  --way_routes_npz "${WAY_ROUTES_IN}" \
  --bad_routes_json "${BAD_JSON}" \
  --out_npz "${WAY_ROUTES_OUT}" \
  --out_report_json "${REPORT_JSON}"
echo ""

echo ">>> [Step 2/2] Building OD-disjoint split (min_hops=3, max_way_len=160)..."
python -m src.data.way_graph.od_disjoint_split \
  --way_routes_npz "${WAY_ROUTES_OUT}" \
  --out_json "${SPLIT_JSON}" \
  --min_hops 3 \
  --max_way_len 160 \
  --seed 0 \
  --val_ratio 0.10 \
  --test_ratio 0.10 \
  --no_per_city

echo ""
echo "======================================"
echo "Done."
echo "======================================"
echo "Strict way_routes: ${WAY_ROUTES_OUT}"
echo "Report:           ${REPORT_JSON}"
echo "Split:            ${SPLIT_JSON}"


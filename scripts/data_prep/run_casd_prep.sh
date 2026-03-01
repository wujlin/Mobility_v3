#!/usr/bin/env bash
# ==========================================
# CASD (Corridor-Aware Segment Diffusion)
# Data Preparation Pipeline (v0, KISS)
# ==========================================
#
# This script builds:
#  1) segment_graph.npz  (collapse degree-2 chains on raster road graph)
#  2) segments_graph_routes.npz (map node_seq -> segment_id sequence per route)
#
# Designed to run on workstation (wsA) with env vars:
#   RAW_ROOT=/home/jinlin/data/geoexplicit_data
#   IN_DATA=$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0
#
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
IN_DATA="${IN_DATA:-${RAW_ROOT%/}/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0}"
PATHS_NPZ="${PATHS_NPZ:-${IN_DATA%/}/paths_graph_combo.npz}"
ROAD_NPZ="${ROAD_NPZ:-${IN_DATA%/}/road_graph_combo.npz}"

OUT_BASE="${OUT_BASE:-${RAW_ROOT%/}/experiments/icml2026_routegen/CASD0_segdata_combo_seed0_term}"
SEED="${SEED:-0}"
SEG_MODE="${SEG_MODE:-collapse}"  # collapse | edge
SEG_COLLAPSE_TIER_MAX="${SEG_COLLAPSE_TIER_MAX:-3}"   # native 推荐 1（忽略 service/unclassified 分支）
SEG_COLLAPSE_DEGREE_MODE="${SEG_COLLAPSE_DEGREE_MODE:-out}"  # native 推荐 undir

echo "Resolved Paths:"
echo "  PATHS_NPZ=${PATHS_NPZ}"
echo "  ROAD_NPZ=${ROAD_NPZ}"
echo "  OUT_BASE=${OUT_BASE}"
echo "  SEG_MODE=${SEG_MODE}"
echo "  SEG_COLLAPSE_TIER_MAX=${SEG_COLLAPSE_TIER_MAX}"
echo "  SEG_COLLAPSE_DEGREE_MODE=${SEG_COLLAPSE_DEGREE_MODE}"
for f in "${PATHS_NPZ}" "${ROAD_NPZ}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing file: ${f}" >&2
    exit 2
  fi
done

mkdir -p "${OUT_BASE}"

echo "======================================"
echo "Step 1: Build segment_graph.npz"
echo "======================================"
mkdir -p "${OUT_BASE}/S1_segment_graph"
PYTHONUNBUFFERED=1 python -u -m src.data.road_graph.build_segment_graph_from_road_graph_npz \
  --road_graph_npz "${ROAD_NPZ}" \
  --paths_graph_npz "${PATHS_NPZ}" \
  --out_dir "${OUT_BASE}/S1_segment_graph" \
  --mode "${SEG_MODE}" \
  --collapse_tier_max "${SEG_COLLAPSE_TIER_MAX}" \
  --collapse_degree_mode "${SEG_COLLAPSE_DEGREE_MODE}" \
  |& tee "${OUT_BASE}/S1_segment_graph/run.log"

SEG_GRAPH="${OUT_BASE}/S1_segment_graph/segment_graph.npz"

echo ""
echo "======================================"
echo "Step 2: Dump per-route segment sequences"
echo "======================================"
mkdir -p "${OUT_BASE}/S2_segment_routes"
PYTHONUNBUFFERED=1 python -u -m src.data.road_graph.dump_segment_sequences_from_paths_graph_npz \
  --paths_graph_npz "${PATHS_NPZ}" \
  --road_graph_npz "${ROAD_NPZ}" \
  --segment_graph_npz "${SEG_GRAPH}" \
  --out_dir "${OUT_BASE}/S2_segment_routes" \
  |& tee "${OUT_BASE}/S2_segment_routes/run.log"

echo ""
echo "======================================"
echo "CASD prep complete!"
echo "======================================"
echo "OUT_BASE=${OUT_BASE}"

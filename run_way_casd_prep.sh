#!/usr/bin/env bash
# ==========================================
# Way-CASD (Way-token CASD)
# Data Preparation Pipeline (v0, KISS)
# ==========================================
#
# This script builds:
#  1) way_routes.npz        (from WorldTrace segments_with_wayid.parquet)
#  2) way_graph.npz         (adjacency CSR from transitions)
#  3) way_features.npz      (tier/length/center/dir from OSM .pbf via pyrosm)
#  4) way_routes_labeled.npz (corridor_type from way_tier)
#
# Designed to run on workstation (wsA) with env vars:
#   RAW_ROOT=/home/jinlin/data/geoexplicit_data
#   SEGMENTS_PARQUET=$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet
#   SEMANTIC_DIR=$RAW_ROOT/worldtrace/detroit_core_v1
#   OSM_PBF=$RAW_ROOT/osm/michigan-latest.osm.pbf
#
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
EXP_ROOT="${EXP_ROOT:-${RAW_ROOT%/}/experiments/icml2026_routegen}"
SEGMENTS_PARQUET="${SEGMENTS_PARQUET:-${RAW_ROOT%/}/worldtrace/detroit_core_v1/segments_with_wayid.parquet}"
SEMANTIC_DIR="${SEMANTIC_DIR:-${RAW_ROOT%/}/worldtrace/detroit_core_v1}"
OSM_PBF="${OSM_PBF:-${RAW_ROOT%/}/osm/michigan-latest.osm.pbf}"

OUT_BASE="${OUT_BASE:-${EXP_ROOT%/}/WAYCASD0_waydata_detroit_seed0}"
ROUTE_CITY="${ROUTE_CITY:-0}"

echo "Resolved Paths:"
echo "  SEGMENTS_PARQUET=${SEGMENTS_PARQUET}"
echo "  SEMANTIC_DIR=${SEMANTIC_DIR}"
echo "  OSM_PBF=${OSM_PBF}"
echo "  OUT_BASE=${OUT_BASE}"
for f in "${SEGMENTS_PARQUET}" "${OSM_PBF}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing file: ${f}" >&2
    exit 2
  fi
done
if [[ ! -f "${SEMANTIC_DIR%/}/osm_road_prob_meta.json" ]]; then
  echo "ERROR: missing bbox meta: ${SEMANTIC_DIR%/}/osm_road_prob_meta.json" >&2
  exit 2
fi

mkdir -p "${OUT_BASE}"

echo "======================================"
echo "Step 1: Build way_routes.npz"
echo "======================================"
mkdir -p "${OUT_BASE}/W1_way_routes"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.build_way_routes_from_segments_parquet \
  --segments_parquet "${SEGMENTS_PARQUET}" \
  --out_npz "${OUT_BASE}/W1_way_routes/way_routes.npz" \
  --route_city "${ROUTE_CITY}" \
  |& tee "${OUT_BASE}/W1_way_routes/run.log"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.audit_way_routes_npz \
  --routes_npz "${OUT_BASE}/W1_way_routes/way_routes.npz" \
  |& tee "${OUT_BASE}/W1_way_routes/audit.log"

echo ""
echo "======================================"
echo "Step 2: Build way_graph.npz"
echo "======================================"
mkdir -p "${OUT_BASE}/W2_way_graph"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.build_way_graph_from_way_routes_npz \
  --way_routes_npz "${OUT_BASE}/W1_way_routes/way_routes.npz" \
  --out_npz "${OUT_BASE}/W2_way_graph/way_graph.npz" \
  |& tee "${OUT_BASE}/W2_way_graph/run.log"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.audit_way_graph_npz \
  --way_graph_npz "${OUT_BASE}/W2_way_graph/way_graph.npz" \
  |& tee "${OUT_BASE}/W2_way_graph/audit.log"

echo ""
echo "======================================"
echo "Step 3: Build way_features.npz"
echo "======================================"
mkdir -p "${OUT_BASE}/W3_way_features"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.build_way_features_from_osm_pbf \
  --osm_pbf "${OSM_PBF}" \
  --semantic_dir "${SEMANTIC_DIR}" \
  --way_routes_npz "${OUT_BASE}/W1_way_routes/way_routes.npz" \
  --out_npz "${OUT_BASE}/W3_way_features/way_features.npz" \
  |& tee "${OUT_BASE}/W3_way_features/run.log"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.audit_way_features_npz \
  --way_features_npz "${OUT_BASE}/W3_way_features/way_features.npz" \
  |& tee "${OUT_BASE}/W3_way_features/audit.log"

echo ""
echo "======================================"
echo "Step 4: Label corridor_type"
echo "======================================"
mkdir -p "${OUT_BASE}/W4_way_routes_labeled"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.label_corridor_type_from_way_features \
  --way_routes_npz "${OUT_BASE}/W1_way_routes/way_routes.npz" \
  --way_features_npz "${OUT_BASE}/W3_way_features/way_features.npz" \
  --out_npz "${OUT_BASE}/W4_way_routes_labeled/way_routes_labeled.npz" \
  --dominant_thr 0.5 \
  |& tee "${OUT_BASE}/W4_way_routes_labeled/run.log"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.audit_way_routes_npz \
  --routes_npz "${OUT_BASE}/W4_way_routes_labeled/way_routes_labeled.npz" \
  |& tee "${OUT_BASE}/W4_way_routes_labeled/audit.log"

echo ""
echo "======================================"
echo "Way-CASD prep complete!"
echo "======================================"
echo "OUT_BASE=${OUT_BASE}"


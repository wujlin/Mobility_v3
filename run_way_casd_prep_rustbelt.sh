#!/usr/bin/env bash
# ==========================================
# Way-CASD Multi-City Data Preparation
# (Detroit + Columbus, Rust Belt)
# ==========================================
#
# This script:
#  0) Generates segments_with_wayid.parquet for Columbus (if not exists)
#  1) Builds per-city way_routes.npz
#  2) Merges into unified way_routes.npz
#  3) Builds unified way_graph.npz (from transitions)
#  4) Builds unified way_features.npz (from OSM pbf)
#  5) Labels corridor_type
#
# Prerequisites:
#   - Detroit already has segments_with_wayid.parquet
#   - OSM pbf files: michigan-latest.osm.pbf, ohio-latest.osm.pbf
#
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
EXP_ROOT="${EXP_ROOT:-${RAW_ROOT%/}/experiments/icml2026_routegen}"
OUT_BASE="${OUT_BASE:-${EXP_ROOT%/}/WAYCASD1_waydata_rustbelt_seed0}"
WAY_GRAPH_UNDIR="${WAY_GRAPH_UNDIR:-0}"  # 0=directed (default), 1=add reverse edges from GT transitions

# Segment extraction knobs (WorldTrace → segments_with_wayid.parquet)
SEG_NUM_WORKERS="${SEG_NUM_WORKERS:-24}"
SEG_CHUNK_SIZE="${SEG_CHUNK_SIZE:-5000}"
SEG_MP_START="${SEG_MP_START:-fork}"

# ===== City Configurations =====
# Detroit (city_id=0)
DETROIT_SEGMENTS="${RAW_ROOT%/}/worldtrace/detroit_core_v1/segments_with_wayid.parquet"
DETROIT_SEMANTIC="${RAW_ROOT%/}/worldtrace/detroit_core_v1"
DETROIT_OSM_PBF="${RAW_ROOT%/}/osm/michigan-latest.osm.pbf"
DETROIT_BBOX="-83.25 42.25 -82.95 42.50"  # min_lon min_lat max_lon max_lat

# Columbus (city_id=1)
COLUMBUS_SEGMENTS="${RAW_ROOT%/}/worldtrace/columbus_core_v1/segments_with_wayid.parquet"
COLUMBUS_SEMANTIC="${RAW_ROOT%/}/worldtrace/columbus_core_v1"
COLUMBUS_OSM_PBF="${RAW_ROOT%/}/osm/ohio-latest.osm.pbf"
COLUMBUS_BBOX="-83.14572187394832 39.84858738738738 -82.85187812605169 40.07381261261261"
COLUMBUS_META="${COLUMBUS_SEMANTIC}/osm_road_prob_meta.json"

# WorldTrace source
TRAJECTORY_ZIP="${RAW_ROOT%/}/worldtrace/OpenTrace_WorldTrace/Trajectory.zip"

echo "======================================"
echo "Way-CASD Multi-City Prep (Rust Belt)"
echo "======================================"
echo "OUT_BASE=${OUT_BASE}"
echo "WAY_GRAPH_UNDIR=${WAY_GRAPH_UNDIR}"
mkdir -p "${OUT_BASE}"

# Ensure Columbus bbox meta exists (used by bbox_from_meta + feature extraction).
if [[ ! -f "${COLUMBUS_META}" ]]; then
    echo "Creating Columbus bbox meta: ${COLUMBUS_META}"
    mkdir -p "${COLUMBUS_SEMANTIC}"
    cat > "${COLUMBUS_META}" << EOF
{
  "grid": {
    "H": 1024,
    "W": 1024,
    "bbox": {
      "min_lon": -83.14572187394832,
      "min_lat": 39.84858738738738,
      "max_lon": -82.85187812605169,
      "max_lat": 40.07381261261261
    }
  }
}
EOF
fi

# ===== Step 0: Generate Columbus segments_with_wayid.parquet (if needed) =====
if [[ ! -f "${COLUMBUS_SEGMENTS}" ]]; then
    echo ""
    echo "======================================"
    echo "Step 0: Build Columbus segments_with_wayid.parquet"
    echo "======================================"
    mkdir -p "${OUT_BASE}/A0_build_columbus_segments"
    mkdir -p "$(dirname "${COLUMBUS_SEGMENTS}")"
    PYTHONUNBUFFERED=1 python -u -m src.data.worldtrace.build_detroit_segments \
        --trajectory_zip "${TRAJECTORY_ZIP}" \
        --out_parquet "${COLUMBUS_SEGMENTS}" \
        --bbox_from_meta "${COLUMBUS_META}" \
        --require_way_id \
        --num_workers "${SEG_NUM_WORKERS}" --chunk_size "${SEG_CHUNK_SIZE}" --mp_start "${SEG_MP_START}" \
        |& tee "${OUT_BASE}/A0_build_columbus_segments/run.log"
else
    echo "[SKIP] Columbus segments_with_wayid.parquet already exists: ${COLUMBUS_SEGMENTS}"
fi

# ===== Step 0b: Way-seq stats (quick sanity) =====
echo ""
echo "======================================"
echo "Step 0b: Way-seq stats (Detroit / Columbus)"
echo "======================================"
mkdir -p "${OUT_BASE}/A1_wayseq_detroit"
PYTHONUNBUFFERED=1 python -u -m src.data.worldtrace.way_seq_stats_from_segments \
    --segments_parquet "${DETROIT_SEGMENTS}" \
    --out_json "${OUT_BASE}/A1_wayseq_detroit/report.json" \
    |& tee "${OUT_BASE}/A1_wayseq_detroit/run.log"
mkdir -p "${OUT_BASE}/A2_wayseq_columbus"
PYTHONUNBUFFERED=1 python -u -m src.data.worldtrace.way_seq_stats_from_segments \
    --segments_parquet "${COLUMBUS_SEGMENTS}" \
    --out_json "${OUT_BASE}/A2_wayseq_columbus/report.json" \
    |& tee "${OUT_BASE}/A2_wayseq_columbus/run.log"

# ===== Step 1a: Build Detroit way_routes.npz =====
echo ""
echo "======================================"
echo "Step 1a: Build Detroit way_routes.npz"
echo "======================================"
mkdir -p "${OUT_BASE}/W1a_detroit_routes"
if [[ ! -f "${DETROIT_SEGMENTS}" ]]; then
    echo "ERROR: Detroit segments not found: ${DETROIT_SEGMENTS}" >&2
    exit 2
fi
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.build_way_routes_from_segments_parquet \
    --segments_parquet "${DETROIT_SEGMENTS}" \
    --out_npz "${OUT_BASE}/W1a_detroit_routes/way_routes.npz" \
    --route_city 0 \
    |& tee "${OUT_BASE}/W1a_detroit_routes/run.log"

# ===== Step 1b: Build Columbus way_routes.npz =====
echo ""
echo "======================================"
echo "Step 1b: Build Columbus way_routes.npz"
echo "======================================"
mkdir -p "${OUT_BASE}/W1b_columbus_routes"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.build_way_routes_from_segments_parquet \
    --segments_parquet "${COLUMBUS_SEGMENTS}" \
    --out_npz "${OUT_BASE}/W1b_columbus_routes/way_routes.npz" \
    --route_city 1 \
    |& tee "${OUT_BASE}/W1b_columbus_routes/run.log"

# ===== Step 2: Merge way_routes.npz =====
echo ""
echo "======================================"
echo "Step 2: Merge way_routes (Detroit + Columbus)"
echo "======================================"
mkdir -p "${OUT_BASE}/W2_merged_routes"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.merge_way_routes_multi_city \
    --inputs "${OUT_BASE}/W1a_detroit_routes/way_routes.npz" \
             "${OUT_BASE}/W1b_columbus_routes/way_routes.npz" \
    --route_cities 0 1 \
    --out_npz "${OUT_BASE}/W2_merged_routes/way_routes.npz" \
    |& tee "${OUT_BASE}/W2_merged_routes/run.log"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.audit_way_routes_npz \
    --routes_npz "${OUT_BASE}/W2_merged_routes/way_routes.npz" \
    |& tee "${OUT_BASE}/W2_merged_routes/audit.log"

# ===== Step 3: Build unified way_graph.npz =====
echo ""
echo "======================================"
echo "Step 3: Build unified way_graph.npz"
echo "======================================"
mkdir -p "${OUT_BASE}/W3_way_graph"
WAY_GRAPH_UNDIR_FLAG=""
if [[ "${WAY_GRAPH_UNDIR}" == "1" ]]; then
  WAY_GRAPH_UNDIR_FLAG="--make_undirected"
fi
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.build_way_graph_from_way_routes_npz \
    --way_routes_npz "${OUT_BASE}/W2_merged_routes/way_routes.npz" \
    --out_npz "${OUT_BASE}/W3_way_graph/way_graph.npz" \
    ${WAY_GRAPH_UNDIR_FLAG} \
    |& tee "${OUT_BASE}/W3_way_graph/run.log"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.audit_way_graph_npz \
    --way_graph_npz "${OUT_BASE}/W3_way_graph/way_graph.npz" \
    |& tee "${OUT_BASE}/W3_way_graph/audit.log"

# ===== Step 4: Build unified way_features.npz =====
# NOTE: This requires OSM pbf for BOTH cities.
# We process each city separately and merge (since pyrosm bbox is per-file).
echo ""
echo "======================================"
echo "Step 4a: Build Detroit way_features"
echo "======================================"
mkdir -p "${OUT_BASE}/W4a_detroit_features"
if [[ ! -f "${DETROIT_OSM_PBF}" ]]; then
    echo "ERROR: Detroit OSM pbf not found: ${DETROIT_OSM_PBF}" >&2
    echo "Download from: https://download.geofabrik.de/north-america/us/michigan-latest.osm.pbf" >&2
    exit 2
fi
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.build_way_features_from_osm_pbf \
    --osm_pbf "${DETROIT_OSM_PBF}" \
    --semantic_dir "${DETROIT_SEMANTIC}" \
    --way_routes_npz "${OUT_BASE}/W2_merged_routes/way_routes.npz" \
    --out_npz "${OUT_BASE}/W4a_detroit_features/way_features.npz" \
    |& tee "${OUT_BASE}/W4a_detroit_features/run.log"

echo ""
echo "======================================"
echo "Step 4b: Build Columbus way_features"
echo "======================================"
mkdir -p "${OUT_BASE}/W4b_columbus_features"
if [[ ! -f "${COLUMBUS_OSM_PBF}" ]]; then
    echo "ERROR: Columbus OSM pbf not found: ${COLUMBUS_OSM_PBF}" >&2
    echo "Download from: https://download.geofabrik.de/north-america/us/ohio-latest.osm.pbf" >&2
    exit 2
fi
# Need to create Columbus osm_road_prob_meta.json if not exists
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.build_way_features_from_osm_pbf \
    --osm_pbf "${COLUMBUS_OSM_PBF}" \
    --semantic_dir "${COLUMBUS_SEMANTIC}" \
    --way_routes_npz "${OUT_BASE}/W2_merged_routes/way_routes.npz" \
    --out_npz "${OUT_BASE}/W4b_columbus_features/way_features.npz" \
    |& tee "${OUT_BASE}/W4b_columbus_features/run.log"

# Merge features from both cities
echo ""
echo "======================================"
echo "Step 4c: Merge way_features (Detroit + Columbus)"
echo "======================================"
mkdir -p "${OUT_BASE}/W4_way_features"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.merge_way_features_multi_city \
    --inputs "${OUT_BASE}/W4a_detroit_features/way_features.npz" \
             "${OUT_BASE}/W4b_columbus_features/way_features.npz" \
    --way_routes_npz "${OUT_BASE}/W2_merged_routes/way_routes.npz" \
    --out_npz "${OUT_BASE}/W4_way_features/way_features.npz" \
    |& tee "${OUT_BASE}/W4_way_features/run.log"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.audit_way_features_npz \
    --way_features_npz "${OUT_BASE}/W4_way_features/way_features.npz" \
    |& tee "${OUT_BASE}/W4_way_features/audit.log"

# ===== Step 5: Label corridor_type =====
echo ""
echo "======================================"
echo "Step 5: Label corridor_type"
echo "======================================"
mkdir -p "${OUT_BASE}/W5_way_routes_labeled"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.label_corridor_type_from_way_features \
    --way_routes_npz "${OUT_BASE}/W2_merged_routes/way_routes.npz" \
    --way_features_npz "${OUT_BASE}/W4_way_features/way_features.npz" \
    --out_npz "${OUT_BASE}/W5_way_routes_labeled/way_routes_labeled.npz" \
    --dominant_thr 0.5 \
    |& tee "${OUT_BASE}/W5_way_routes_labeled/run.log"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.audit_way_routes_npz \
    --routes_npz "${OUT_BASE}/W5_way_routes_labeled/way_routes_labeled.npz" \
    |& tee "${OUT_BASE}/W5_way_routes_labeled/audit.log"

echo ""
echo "======================================"
echo "Way-CASD Multi-City Prep Complete!"
echo "======================================"
echo "OUT_BASE=${OUT_BASE}"
echo ""
echo "Next steps:"
echo "  # Train AE:"
echo "  python -m src.training.train_way_casd_autoencoder \\"
echo "    --way_routes_npz \"\$OUT_BASE/W5_way_routes_labeled/way_routes_labeled.npz\" \\"
echo "    --way_graph_npz \"\$OUT_BASE/W3_way_graph/way_graph.npz\" \\"
echo "    --way_features_npz \"\$OUT_BASE/W4_way_features/way_features.npz\" \\"
echo "    --out_dir \"\$OUT_BASE/W6_train_ae\" \\"
echo "    --batch_size 512 --num_workers 24 --n_epochs 30 --d_model 256 --n_latent 32 --max_candidates 64"

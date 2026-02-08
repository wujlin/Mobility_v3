#!/usr/bin/env bash
# ==========================================
# Porto Taxi: 完整预处理 Pipeline
# ==========================================
#
# 两阶段：
#   Phase 0: CSV → segments_with_wayid.parquet (Valhalla map matching)
#   Phase 1: 复用 run_way_casd_prep.sh 走标准 pipeline
#
# 前置条件：
#   1. train.csv 已存在于 $RAW_ROOT/porto_taxi/raw/
#   2. portugal-latest.osm.pbf 已存在于 $RAW_ROOT/osm/
#   3. Valhalla Docker 已在 localhost:8002 运行 (用 porto_extract.osm.pbf 构建的 tiles)
#
# 用法 (在工作站项目根目录下):
#   # 调试 100 条
#   bash tools/porto/run_porto_prep.sh --limit 100
#
#   # 全量
#   bash tools/porto/run_porto_prep.sh
#
set -euo pipefail

# ── 参数 ──
LIMIT="${1:-0}"  # 默认全量; 传 --limit 100 改为调试
if [[ "${LIMIT}" == "--limit" ]]; then
    LIMIT="${2:-100}"
fi

# ── 路径 ──
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

PORTO_CSV="${RAW_ROOT}/porto_taxi/raw/train.csv"
PORTO_PARQUET="${RAW_ROOT}/porto_taxi/segments_with_wayid.parquet"
PORTO_BBOX_META="${PROJ_ROOT}/tools/porto/porto_bbox_meta.json"
# 用 osmium 裁剪后的 porto_extract (16MB) 而非全国 PBF (382MB)
# pyrosm 需要完整解析 PBF，全国文件极慢
OSM_PBF="${RAW_ROOT}/osm/porto_extract.osm.pbf"
VALHALLA_URL="${VALHALLA_URL:-http://localhost:8002}"
WORKERS="${WORKERS:-8}"

# Porto 的 semantic_dir: 需要包含 osm_road_prob_meta.json
# 我们把 porto_bbox_meta.json 软链接过去
PORTO_SEMANTIC_DIR="${RAW_ROOT}/porto_taxi/semantic"
mkdir -p "${PORTO_SEMANTIC_DIR}"
if [[ ! -f "${PORTO_SEMANTIC_DIR}/osm_road_prob_meta.json" ]]; then
    cp "${PORTO_BBOX_META}" "${PORTO_SEMANTIC_DIR}/osm_road_prob_meta.json"
    echo "已创建 ${PORTO_SEMANTIC_DIR}/osm_road_prob_meta.json"
fi

echo "======================================"
echo "Porto Taxi 预处理 Pipeline"
echo "======================================"
echo "  RAW_ROOT=${RAW_ROOT}"
echo "  CSV=${PORTO_CSV}"
echo "  OSM_PBF=${OSM_PBF}"
echo "  PARQUET=${PORTO_PARQUET}"
echo "  VALHALLA=${VALHALLA_URL}"
echo "  WORKERS=${WORKERS}"
echo "  LIMIT=${LIMIT}"
echo ""

# ── 检查前置文件 ──
for f in "${PORTO_CSV}" "${OSM_PBF}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: 文件不存在: ${f}" >&2
        exit 2
    fi
done

# ── Phase 0: CSV → segments_with_wayid.parquet ──
echo "======================================"
echo "Phase 0: Porto CSV → segments_with_wayid.parquet"
echo "======================================"

if [[ -f "${PORTO_PARQUET}" ]] && [[ "${LIMIT}" == "0" ]]; then
    echo "  已存在 ${PORTO_PARQUET}，跳过 (如需重跑请先删除)"
else
    LIMIT_ARG=""
    if [[ "${LIMIT}" != "0" ]]; then
        LIMIT_ARG="--limit ${LIMIT}"
    fi

    cd "${PROJ_ROOT}"
    PYTHONUNBUFFERED=1 python -u -m tools.porto.porto_csv_to_segments_parquet \
        --csv "${PORTO_CSV}" \
        --out_parquet "${PORTO_PARQUET}" \
        --bbox_meta "${PORTO_BBOX_META}" \
        --valhalla_url "${VALHALLA_URL}" \
        --workers "${WORKERS}" \
        ${LIMIT_ARG} \
        |& tee "${RAW_ROOT}/porto_taxi/map_matching.log"
fi

echo ""
echo "======================================"
echo "Phase 1: 标准 Way-CASD 数据 pipeline"
echo "======================================"

# 复用 run_way_casd_prep.sh，只需覆盖环境变量
export RAW_ROOT
export SEGMENTS_PARQUET="${PORTO_PARQUET}"
export SEMANTIC_DIR="${PORTO_SEMANTIC_DIR}"
export OSM_PBF
export OUT_BASE="${RAW_ROOT}/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
export ROUTE_CITY=0

cd "${PROJ_ROOT}"
bash run_way_casd_prep.sh

echo ""
echo "======================================"
echo "Porto 预处理全部完成!"
echo "======================================"
echo "OUT_BASE=${OUT_BASE}"
echo ""
echo "产出文件:"
echo "  ${OUT_BASE}/W1_way_routes/way_routes.npz"
echo "  ${OUT_BASE}/W2_way_graph/way_graph.npz"
echo "  ${OUT_BASE}/W3_way_features/way_features.npz"
echo "  ${OUT_BASE}/W4_way_routes_labeled/way_routes_labeled.npz"

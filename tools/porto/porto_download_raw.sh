#!/usr/bin/env bash
#
# Porto Taxi (ECML/PKDD 2015) + Portugal OSM PBF downloader for WSA.
# - Prefers Kaggle *competition* source (has extra metadata zips).
# - Falls back to Kaggle *dataset* mirror if competition download fails.
# - Writes ONLY to $RAW_ROOT (keeps repo clean). Avoids `exit` to not kill an interactive shell.
#
# Usage:
#   RAW_ROOT=/home/jinlin/data/geoexplicit_data bash tools/porto/porto_download_raw.sh
#
# Optional overrides:
#   PORTO_BBOX_LBRB="-8.72,41.10,-8.52,41.22"   # lon_min,lat_min,lon_max,lat_max (for osmium extract)
#

RAW_ROOT="${RAW_ROOT:-$HOME/data/geoexplicit_data}"
PORTO_RAW="${RAW_ROOT}/porto_taxi/raw"
OSM_DIR="${RAW_ROOT}/osm"

KAGGLE_COMP_SLUG="pkdd-15-taxi-trip-time-prediction-ii"
KAGGLE_DATASET_SLUG="crailtap/taxi-trajectory"

GEF_PBF_URL="https://download.geofabrik.de/europe/portugal-latest.osm.pbf"
PORTUGAL_PBF="${OSM_DIR}/portugal-latest.osm.pbf"
PORTO_EXTRACT_PBF="${OSM_DIR}/porto_extract.osm.pbf"
PORTO_BBOX_LBRB="${PORTO_BBOX_LBRB:--8.72,41.10,-8.52,41.22}"

_ts() { date +"%Y-%m-%d %H:%M:%S"; }
_info() { echo "[$(_ts)] [INFO] $*"; }
_warn() { echo "[$(_ts)] [WARN] $*" 1>&2; }

_have() { command -v "$1" >/dev/null 2>&1; }

_download_url() {
  # _download_url URL OUT_PATH
  local url="$1"
  local out="$2"

  if [[ -f "$out" ]]; then
    _info "Exists: $out"
    return 0
  fi

  if _have wget; then
    _info "Downloading (wget): $url"
    wget -O "$out" "$url" || return 1
    return 0
  fi
  if _have curl; then
    _info "Downloading (curl): $url"
    curl -L -o "$out" "$url" || return 1
    return 0
  fi

  _warn "Missing downloader: install `wget` or `curl`."
  return 1
}

_unzip_all_in_dir() {
  local dir="$1"
  if ! _have unzip; then
    _warn "Missing `unzip`. Install it first."
    return 1
  fi

  local z
  shopt -s nullglob
  local zips=("$dir"/*.zip)
  shopt -u nullglob

  if (( ${#zips[@]} == 0 )); then
    _info "No .zip files found in: $dir"
    return 0
  fi

  for z in "${zips[@]}"; do
    _info "Unzipping: $(basename "$z")"
    unzip -o "$z" -d "$dir" >/dev/null || return 1
  done
  return 0
}

_train_csv_sanity_ok() {
  local csv="$1"
  if [[ ! -f "$csv" ]]; then
    _warn "Missing: $csv"
    return 1
  fi
  local header
  header="$(head -n 1 "$csv" | tr -d '\r' | tr -d '\n')"
  if [[ "$header" != TRIP_ID* ]]; then
    _warn "Unexpected train.csv header: $header"
    return 1
  fi
  local n
  n="$(wc -l "$csv" | awk '{print $1}')"
  if [[ -z "$n" || "$n" -lt 2 ]]; then
    _warn "train.csv seems empty (wc -l = $n)"
    return 1
  fi
  return 0
}

_try_kaggle_competition() {
  local dir="$1"
  if ! _have kaggle; then
    _warn "Missing `kaggle` CLI. Install via: pip install kaggle"
    _warn "Then configure: ~/.kaggle/kaggle.json (chmod 600)."
    return 1
  fi

  _info "Kaggle competition download: $KAGGLE_COMP_SLUG"
  local out rc
  out="$(kaggle competitions download -c "$KAGGLE_COMP_SLUG" -p "$dir" 2>&1)"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "$out" 1>&2
    if echo "$out" | grep -qiE "403|forbidden|accept.*rules|permission"; then
      _warn "Competition download failed (likely rules not accepted)."
      _warn "Please visit and click 'Accept Rules':"
      _warn "  https://www.kaggle.com/competitions/${KAGGLE_COMP_SLUG}/rules"
    fi
    return 1
  fi
  return 0
}

_try_kaggle_dataset_fallback() {
  local dir="$1"
  if ! _have kaggle; then
    return 1
  fi
  _info "Kaggle dataset fallback download: $KAGGLE_DATASET_SLUG"
  kaggle datasets download -d "$KAGGLE_DATASET_SLUG" -p "$dir" || return 1
  return 0
}

main() {
  _info "RAW_ROOT=$RAW_ROOT"
  _info "PORTO_RAW=$PORTO_RAW"
  _info "OSM_DIR=$OSM_DIR"
  mkdir -p "$PORTO_RAW" "$OSM_DIR"

  local ok_kaggle=0
  _info ">>> [1/3] Downloading Porto Taxi (prefer Kaggle competition, fallback dataset)..."
  if _try_kaggle_competition "$PORTO_RAW"; then
    ok_kaggle=1
  else
    _warn "Competition download failed; trying dataset fallback..."
    if _try_kaggle_dataset_fallback "$PORTO_RAW"; then
      ok_kaggle=1
    else
      _warn "Dataset fallback also failed. Please check Kaggle CLI configuration and access."
    fi
  fi

  if (( ok_kaggle == 1 )); then
    _info ">>> [1.1] Unzipping Kaggle archives..."
    _unzip_all_in_dir "$PORTO_RAW" || _warn "Unzip failed. Check disk space / zip files."
    local train_csv="${PORTO_RAW}/train.csv"
    if _train_csv_sanity_ok "$train_csv"; then
      _info "Sanity OK: train.csv (header + non-empty). Removing .zip to save space..."
      rm -f "$PORTO_RAW"/*.zip
    else
      _warn "Sanity check failed; keeping .zip files. Please inspect: $PORTO_RAW"
    fi
  fi

  _info ">>> [2/3] Downloading Portugal OSM PBF (Geofabrik)..."
  _download_url "$GEF_PBF_URL" "$PORTUGAL_PBF" || _warn "OSM download failed. Check network."

  _info ">>> [2.1] Optional: Extract Porto bbox via `osmium` (seconds, if installed)..."
  if _have osmium && [[ -f "$PORTUGAL_PBF" ]]; then
    if [[ -f "$PORTO_EXTRACT_PBF" ]]; then
      _info "Exists: $PORTO_EXTRACT_PBF"
    else
      _info "osmium extract --bbox $PORTO_BBOX_LBRB -> $PORTO_EXTRACT_PBF"
      # NOTE: bbox order is lon_min,lat_min,lon_max,lat_max
      osmium extract -b "$PORTO_BBOX_LBRB" -o "$PORTO_EXTRACT_PBF" "$PORTUGAL_PBF" \
        || _warn "osmium extract failed; you can retry after installing osmium-tool."
    fi
  else
    _info "Skip porto extract: `osmium` not found (optional)."
  fi

  _info ">>> [3/3] Done. Artifacts:"
  ls -lh "$PORTO_RAW" 2>/dev/null | head -n 50 || true
  ls -lh "$PORTUGAL_PBF" "$PORTO_EXTRACT_PBF" 2>/dev/null || true

  _info "Next steps (not in this script):"
  _info "  - Clean POLYLINE + filter trips -> parquet"
  _info "  - Map matching (Valhalla) -> way sequences"
  _info "  - Build Way-CASD: way_routes.npz / way_graph.npz / way_features.npz"
}

main "$@"


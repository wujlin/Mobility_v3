#!/bin/bash
# Porto checklist_exp follow-up:
# 1) RL dense sched e20: K16 + dest_efficient + dump_way_seqs
# 2) P1 stepemb e100:    K16 + dest_efficient + dump_way_seqs
# 3) OD coverage/diversity: P1 vs RL (optional + baselines)
#
# Notes:
# - This script is incremental: existing outputs are skipped.
# - It does NOT use `set -e` to avoid abrupt terminal exits.
# - Override paths/knobs via env vars when needed.

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR" || true

# -----------------------------
# Data paths (override via env)
# -----------------------------
EXP_ROOT="${EXP_ROOT:-/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen}"
DATA_BASE="${DATA_BASE:-${EXP_ROOT}/WAYCASD0_waydata_porto_seed0}"
WAY_ROUTES="${WAY_ROUTES:-${DATA_BASE}/W5_way_routes_strict_gate/way_routes_strict_gate.npz}"
WAY_GRAPH="${WAY_GRAPH:-${DATA_BASE}/W2_way_graph/way_graph.npz}"
WAY_FEATURES="${WAY_FEATURES:-${DATA_BASE}/W3_way_features/way_features.npz}"
WAY_REGIONS="${WAY_REGIONS:-${DATA_BASE}/region_sweep/way_regions_louvain_res5_seed0.npz}"
SPLIT_JSON="${SPLIT_JSON:-${DATA_BASE}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json}"
CITY_META="${CITY_META:-/home/jinlin/data/geoexplicit_data/porto_taxi/semantic/osm_road_prob_meta.json}"

# -----------------------------
# Model checkpoints (override)
# -----------------------------
FLOW_CKPT="${FLOW_CKPT:-_sync/wsa/pi_verify/20260212_porto_flow_xattn_regionseq_dev10p_s0/ckpt_best.pt}"
P1_AE_CKPT="${P1_AE_CKPT:-_sync/wsa/pi_verify/20260214_porto_p1_stepemb_cont_e100_s0/ckpt_best.pt}"
RL_AE_CKPT="${RL_AE_CKPT:-_sync/wsa/pi_verify/20260216_porto_rl_dense_sched09to03_e20_freshE100_from_e100_s0/ckpt_best.pt}"

# -----------------------------
# Output dirs
# -----------------------------
OUT_P1="${OUT_P1:-_sync/wsa/pi_verify/20260214_porto_p1_stepemb_cont_e100_s0/eval}"
OUT_RL="${OUT_RL:-_sync/wsa/pi_verify/20260216_porto_rl_dense_sched09to03_e20_freshE100_from_e100_s0/eval}"
OUT_OD="${OUT_OD:-_sync/wsa/pi_verify/20260221_porto_p1_vs_rl_od_eval_s0}"
mkdir -p "$OUT_P1" "$OUT_RL" "$OUT_OD"

# -----------------------------
# Runtime knobs
# -----------------------------
PY_BIN="${PY_BIN:-python}"
N_ROUTES="${N_ROUTES:-5000}"
K="${K:-16}"
SEED="${SEED:-0}"
EVAL_BS="${EVAL_BS:-64}"           # keep default-safe; increase to 96/128 if GPU allows
SHAPE_SCOPE="${SHAPE_SCOPE:-none}" # none/selected/all ; none is faster and enough for Phase C
RUN_BASELINE_OD="${RUN_BASELINE_OD:-0}"  # 1 => include existing RNN/Transformer per_route in OD eval

P1_B_JSON="${OUT_P1}/binned_e100_k16_dest_efficient_n5000_wayseqs.json"
P1_PR_JSON="${OUT_P1}/per_route_e100_k16_dest_efficient_n5000_wayseqs.json"
RL_B_JSON="${OUT_RL}/binned_rl_k16_dest_efficient_n5000.json"
RL_PR_JSON="${OUT_RL}/per_route_rl_k16_dest_efficient_n5000.json"
OD_JSON="${OUT_OD}/od_coverage_diversity_p1_vs_rl_k16_n5000.json"

RNN_PR_DEFAULT="_sync/wsa/pi_verify/20260212_porto_phaseBC_n5000_s0/phaseB/per_route_rnn_beam10_n5000.json"
TR_PR_DEFAULT="_sync/wsa/pi_verify/20260212_porto_phaseBC_n5000_s0/phaseB/per_route_transformer_beam10_n5000.json"

echo "=================================================="
echo "Checklist Follow-up (incremental)"
echo "=================================================="
echo "WAY_ROUTES:   $WAY_ROUTES"
echo "WAY_GRAPH:    $WAY_GRAPH"
echo "WAY_FEATURES: $WAY_FEATURES"
echo "WAY_REGIONS:  $WAY_REGIONS"
echo "SPLIT_JSON:   $SPLIT_JSON"
echo "CITY_META:    $CITY_META"
echo "P1_AE_CKPT:   $P1_AE_CKPT"
echo "RL_AE_CKPT:   $RL_AE_CKPT"
echo "FLOW_CKPT:    $FLOW_CKPT"
echo "N_ROUTES:     $N_ROUTES, K: $K, EVAL_BS: $EVAL_BS, SHAPE_SCOPE: $SHAPE_SCOPE"
echo

run_cmd() {
  local label="$1"
  local logfile="$2"
  shift 2
  echo ">>> [$label] running..."
  "$@" 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!!! [$label] failed with rc=$rc (see $logfile)"
  else
    echo ">>> [$label] done"
  fi
  return "$rc"
}

check_file() {
  local f="$1"
  if [ -f "$f" ]; then
    echo "[OK] $f"
    return 0
  fi
  echo "[MISS] $f"
  return 1
}

has_way_seqs() {
  local f="$1"
  "$PY_BIN" - <<PY
import json,sys
from pathlib import Path
p=Path("$f")
if not p.exists():
    print("NOFILE")
    sys.exit(2)
d=json.loads(p.read_text())
rows=d.get("per_route", [])
if not rows:
    print("EMPTY")
    sys.exit(3)
r=rows[0]
g=r.get("greedy", {})
ok=("gt_way_ids" in r) and ("pred_way_ids" in g)
print("YES" if ok else "NO")
sys.exit(0 if ok else 1)
PY
  return $?
}

echo ">>> 1) Preflight checks"
for f in "$WAY_ROUTES" "$WAY_GRAPH" "$WAY_FEATURES" "$WAY_REGIONS" "$SPLIT_JSON" "$CITY_META" "$FLOW_CKPT" "$P1_AE_CKPT" "$RL_AE_CKPT"; do
  check_file "$f"
done
echo

echo ">>> 2) RL dense sched e20: K16 + dest_efficient + wayseq"
need_rl=1
if check_file "$RL_PR_JSON" >/dev/null 2>&1; then
  if has_way_seqs "$RL_PR_JSON" >/dev/null 2>&1; then
    echo "[SKIP] RL per-route already contains gt_way_ids/pred_way_ids: $RL_PR_JSON"
    need_rl=0
  fi
fi

if [ "$need_rl" -eq 1 ]; then
  run_cmd "RL_K16_dest_efficient" "$OUT_RL/run_eval_rl_k16_dest_efficient_n5000.log" \
    "$PY_BIN" -u -m src.evaluation.way_casd_binned_eval \
      --way_routes_npz "$WAY_ROUTES" \
      --way_graph_npz "$WAY_GRAPH" \
      --way_features_npz "$WAY_FEATURES" \
      --ae_ckpt "$RL_AE_CKPT" \
      --flow_ckpt "$FLOW_CKPT" \
      --way_regions_npz "$WAY_REGIONS" \
      --latent_source flow \
      --split_json "$SPLIT_JSON" --split_part test \
      --n_routes "$N_ROUTES" --min_hops 5 --max_way_len 160 --max_decode_len 160 \
      --n_samples_per_route "$K" --sample_select dest_efficient \
      --decode_max_candidates 0 --decode_candidate_policy first \
      --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
      --shape_scope "$SHAPE_SCOPE" \
      --city_grid_meta "0=$CITY_META" \
      --eval_batch_size "$EVAL_BS" \
      --dump_way_seqs \
      --out_json "$RL_B_JSON" \
      --out_per_route_json "$RL_PR_JSON" \
      --device cuda --seed "$SEED"
fi
echo

echo ">>> 3) P1 e100: K16 + dest_efficient + wayseq"
need_p1=1
if check_file "$P1_PR_JSON" >/dev/null 2>&1; then
  if has_way_seqs "$P1_PR_JSON" >/dev/null 2>&1; then
    echo "[SKIP] P1 per-route already contains gt_way_ids/pred_way_ids: $P1_PR_JSON"
    need_p1=0
  fi
fi

if [ "$need_p1" -eq 1 ]; then
  run_cmd "P1_K16_dest_efficient" "$OUT_P1/run_eval_e100_k16_dest_efficient_n5000_wayseqs.log" \
    "$PY_BIN" -u -m src.evaluation.way_casd_binned_eval \
      --way_routes_npz "$WAY_ROUTES" \
      --way_graph_npz "$WAY_GRAPH" \
      --way_features_npz "$WAY_FEATURES" \
      --ae_ckpt "$P1_AE_CKPT" \
      --flow_ckpt "$FLOW_CKPT" \
      --way_regions_npz "$WAY_REGIONS" \
      --latent_source flow \
      --split_json "$SPLIT_JSON" --split_part test \
      --n_routes "$N_ROUTES" --min_hops 5 --max_way_len 160 --max_decode_len 160 \
      --n_samples_per_route "$K" --sample_select dest_efficient \
      --decode_max_candidates 0 --decode_candidate_policy first \
      --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
      --shape_scope "$SHAPE_SCOPE" \
      --city_grid_meta "0=$CITY_META" \
      --eval_batch_size "$EVAL_BS" \
      --dump_way_seqs \
      --out_json "$P1_B_JSON" \
      --out_per_route_json "$P1_PR_JSON" \
      --device cuda --seed "$SEED"
fi
echo

echo ">>> 4) OD coverage/diversity (P1 vs RL)"
OD_ARGS=(
  --method "P1_E100|greedy=${P1_PR_JSON}"
  --method "RL_DenseSched_E20|greedy=${RL_PR_JSON}"
)
if [ "$RUN_BASELINE_OD" = "1" ]; then
  if [ -f "$RNN_PR_DEFAULT" ] && [ -f "$TR_PR_DEFAULT" ]; then
    OD_ARGS+=( --method "RNN_AR_b10|beam=${RNN_PR_DEFAULT}" )
    OD_ARGS+=( --method "Transformer_AR_b10|beam=${TR_PR_DEFAULT}" )
  else
    echo "[WARN] RUN_BASELINE_OD=1 but baseline per_route files not found; skip baselines."
  fi
fi

run_cmd "OD_Coverage_Diversity" "$OUT_OD/run_od_coverage_diversity_p1_vs_rl_k16_n5000.log" \
  "$PY_BIN" -u -m src.evaluation.od_coverage_diversity_eval \
    "${OD_ARGS[@]}" \
    --out_json "$OD_JSON" \
    --k "$K" \
    --min_routes_per_od 3 \
    --jaccard_threshold 0.5 \
    --save_per_od \
    --way_routes_npz "$WAY_ROUTES" \
    --split_json "$SPLIT_JSON" --split_part test

echo
echo ">>> 5) Quick summary"
"$PY_BIN" - <<PY
import json, math
from pathlib import Path

def read_sr_len(path):
    p=Path(path)
    if not p.exists():
        return None
    d=json.loads(p.read_text())
    g=d.get("overall",{}).get("greedy",{})
    cells=g.get("cells",{})
    if not cells:
        return {"sr": float("nan"), "len_ratio": float("nan"), "succ_len_ratio": float("nan")}
    nsum=0.0
    sr_num=0.0
    lr_num=0.0; lr_den=0.0
    slr_num=0.0; slr_den=0.0
    for c in cells.values():
        n=float(c.get("n",0))
        sr=float(c.get("success_rate", float("nan")))
        if math.isfinite(sr) and n>0:
            sr_num += n*sr; nsum += n
        lr=c.get("len_ratio",{}).get("mean", float("nan")) if isinstance(c.get("len_ratio"),dict) else float("nan")
        if math.isfinite(lr) and n>0:
            lr_num += n*lr; lr_den += n
        slr = c.get("success_only_len_ratio",{}).get("mean", float("nan")) if isinstance(c.get("success_only_len_ratio"),dict) else float("nan")
        n_succ = c.get("success_only_n", 0)
        if isinstance(n_succ,(int,float)) and math.isfinite(float(slr)) and n_succ>0:
            slr_num += float(n_succ)*float(slr); slr_den += float(n_succ)
    return {
        "sr": (sr_num/nsum if nsum>0 else float("nan")),
        "len_ratio": (lr_num/lr_den if lr_den>0 else float("nan")),
        "succ_len_ratio": (slr_num/slr_den if slr_den>0 else float("nan")),
    }

p1 = read_sr_len("$P1_B_JSON")
rl = read_sr_len("$RL_B_JSON")
print("P1:", p1)
print("RL:", rl)
od_path=Path("$OD_JSON")
if od_path.exists():
    od=json.loads(od_path.read_text())
    print("OD summary_table:")
    for row in od.get("summary_table", []):
        print(row)
else:
    print("OD summary missing:", str(od_path))
PY

echo
echo ">>> DONE"
echo "P1 binned: $P1_B_JSON"
echo "P1 per_route: $P1_PR_JSON"
echo "RL binned: $RL_B_JSON"
echo "RL per_route: $RL_PR_JSON"
echo "OD json: $OD_JSON"

#!/usr/bin/env bash

# Porto baseline eval + visualization bundle
# - Reuse existing per_route outputs when present
# - Run missing baseline evals only
# - Produce Phase-C coverage/diversity, success-only quality summary
# - Produce hero OD figure + loop Leaflet + len_ratio histogram
#
# Run in WSL (with conda env `dpl`) for full plotting support.

set -u

PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJ_ROOT" || exit 1

DATA_BASE="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${DATA_BASE}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${DATA_BASE}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_BASE}/W3_way_features/way_features.npz"
SPLIT_JSON="${DATA_BASE}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"
CITY_META="/home/jinlin/data/geoexplicit_data/porto_taxi/semantic/osm_road_prob_meta.json"

RNN_CKPT="_sync/wsa/pi_verify/20260210_porto_phase1_s0/B2_rnn_ar_dev10p/ckpt_best.pt"
TR_CKPT="_sync/wsa/pi_verify/20260210_porto_phase1_s0/B3_transformer_ar_dev10p/ckpt_best.pt"

# Main method (already evaluated): BetaVAE64 + FlowMu + K16 + anti-loop
B2_PER="_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B2_eval_k16_antiloop/per_route_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json"

OUT_ROOT="_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0"
OUT_BASE="${OUT_ROOT}/baseline_eval"
OUT_PHASEC="${OUT_ROOT}/phaseC"
OUT_VIZ="${OUT_ROOT}/viz"
mkdir -p "$OUT_BASE" "$OUT_PHASEC" "$OUT_VIZ"

RNN_PER="${OUT_BASE}/per_route_rnn_beam10_n5000.json"
RNN_BIN="${OUT_BASE}/binned_rnn_beam10_n5000.json"
TR_PER="${OUT_BASE}/per_route_transformer_beam10_n5000.json"
TR_BIN="${OUT_BASE}/binned_transformer_beam10_n5000.json"

if [ -s "_sync/wsa/pi_verify/20260221_porto_baseline_quality_n5000_s0/per_route_rnn_beam10_n5000.json" ] && [ ! -s "$RNN_PER" ]; then
  cp "_sync/wsa/pi_verify/20260221_porto_baseline_quality_n5000_s0/per_route_rnn_beam10_n5000.json" "$RNN_PER"
  cp "_sync/wsa/pi_verify/20260221_porto_baseline_quality_n5000_s0/binned_rnn_beam10_n5000.json" "$RNN_BIN"
fi
if [ -s "_sync/wsa/pi_verify/20260221_porto_baseline_quality_n5000_s0/per_route_transformer_beam10_n5000.json" ] && [ ! -s "$TR_PER" ]; then
  cp "_sync/wsa/pi_verify/20260221_porto_baseline_quality_n5000_s0/per_route_transformer_beam10_n5000.json" "$TR_PER"
  cp "_sync/wsa/pi_verify/20260221_porto_baseline_quality_n5000_s0/binned_transformer_beam10_n5000.json" "$TR_BIN"
fi

FAILED=()

run_step() {
  local name="$1"
  shift
  echo ""
  echo "======================================================================"
  echo ">>> [${name}] START"
  echo "======================================================================"
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo ">>> [${name}] FAILED (rc=${rc})"
    FAILED+=("${name}")
  else
    echo ">>> [${name}] OK"
  fi
  echo "======================================================================"
}

# Step 1: Missing baseline evals only
if [ ! -s "$RNN_PER" ] || [ ! -s "$RNN_BIN" ]; then
  run_step "EVAL_RNN_B10_N5000" \
    conda run -n dpl python -u -m src.evaluation.unified_binned_eval \
      --method rnn_ar \
      --ckpt "$RNN_CKPT" \
      --way_routes_npz "$WAY_ROUTES" \
      --way_graph_npz "$WAY_GRAPH" \
      --way_features_npz "$WAY_FEATURES" \
      --split_json "$SPLIT_JSON" \
      --split_part test \
      --city_grid_meta "0=$CITY_META" \
      --n_routes 5000 \
      --min_hops 5 \
      --max_way_len 160 \
      --max_decode_len 160 \
      --beam_size 10 \
      --decode_max_candidates -1 \
      --eval_batch_size 256 \
      --dump_way_seqs \
      --out_json "$RNN_BIN" \
      --out_per_route_json "$RNN_PER" \
      --device cuda \
      --seed 0
else
  echo ">>> [EVAL_RNN_B10_N5000] SKIP (reuse existing: $RNN_PER)"
fi

if [ ! -s "$TR_PER" ] || [ ! -s "$TR_BIN" ]; then
  run_step "EVAL_TRANSFORMER_B10_N5000" \
    conda run -n dpl python -u -m src.evaluation.unified_binned_eval \
      --method transformer_ar \
      --ckpt "$TR_CKPT" \
      --way_routes_npz "$WAY_ROUTES" \
      --way_graph_npz "$WAY_GRAPH" \
      --way_features_npz "$WAY_FEATURES" \
      --split_json "$SPLIT_JSON" \
      --split_part test \
      --city_grid_meta "0=$CITY_META" \
      --n_routes 5000 \
      --min_hops 5 \
      --max_way_len 160 \
      --max_decode_len 160 \
      --beam_size 10 \
      --decode_max_candidates -1 \
      --eval_batch_size 256 \
      --dump_way_seqs \
      --out_json "$TR_BIN" \
      --out_per_route_json "$TR_PER" \
      --device cuda \
      --seed 0
else
  echo ">>> [EVAL_TRANSFORMER_B10_N5000] SKIP (reuse existing: $TR_PER)"
fi

# Step 2: Phase-C coverage/diversity
PHASEC_JSON="${OUT_PHASEC}/od_coverage_diversity_b2_rnn_tr_k16_n5000.json"
run_step "PHASEC_B2_RNN_TR_K16" \
  conda run -n dpl python -u -m src.evaluation.od_coverage_diversity_eval \
    --method "BetaVAE64_FlowMu_K16_AL|greedy=${B2_PER}" \
    --method "RNN_AR_b10|beam=${RNN_PER}" \
    --method "Transformer_AR_b10|beam=${TR_PER}" \
    --k 16 \
    --min_routes_per_od 3 \
    --jaccard_threshold 0.3 \
    --save_per_od \
    --out_json "$PHASEC_JSON"

# Step 3: success-only quality summary
run_step "SUCCESS_ONLY_SUMMARY_B2_RNN_TR" \
  conda run -n dpl python -u - <<'PY'
import json
from pathlib import Path
import numpy as np

out = Path("_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/success_only_quality_b2_vs_baselines_n5000.json")
out.parent.mkdir(parents=True, exist_ok=True)
methods = {
    "BetaVAE64_FlowMu_K16_AL": (
        "_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B2_eval_k16_antiloop/per_route_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json",
        "greedy",
    ),
    "RNN_AR_b10": (
        "_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/baseline_eval/per_route_rnn_beam10_n5000.json",
        "beam",
    ),
    "Transformer_AR_b10": (
        "_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/baseline_eval/per_route_transformer_beam10_n5000.json",
        "beam",
    ),
}

def stat(xs):
    arr = np.asarray([float(x) for x in xs if np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }

res = {}
for name, (path, decode) in methods.items():
    rows = json.loads(Path(path).read_text(encoding="utf-8")).get("per_route", [])
    succ_flags = []
    hit_wall = []
    dead_end = []
    succ_loop = []
    succ_lenr = []
    succ_jacc = []
    for r in rows:
        d = r.get(decode, {})
        s = bool(d.get("success", False))
        succ_flags.append(1.0 if s else 0.0)
        hit_wall.append(1.0 if bool(d.get("hit_wall", False)) else 0.0)
        dead_end.append(1.0 if bool(d.get("dead_end", False)) else 0.0)
        if s:
            succ_loop.append(1.0 if bool(d.get("has_loop", False)) else 0.0)
            succ_lenr.append(float(d.get("len_ratio", float("nan"))))
            succ_jacc.append(float(d.get("jaccard", float("nan"))))
    res[name] = {
        "n_routes": int(len(rows)),
        "success_rate": float(np.mean(succ_flags)) if succ_flags else float("nan"),
        "hit_wall_rate": float(np.mean(hit_wall)) if hit_wall else float("nan"),
        "dead_end_rate": float(np.mean(dead_end)) if dead_end else float("nan"),
        "success_only_n": int(sum(1 for x in succ_flags if x > 0)),
        "success_only_loop_rate": float(np.mean(succ_loop)) if succ_loop else float("nan"),
        "success_only_len_ratio": stat(succ_lenr),
        "success_only_jaccard": stat(succ_jacc),
    }

obj = {"ok": True, "task": "success_only_quality_summary", "methods": res}
out.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] saved: {out}")
for k, v in res.items():
    print(
        f"{k}: succ={v['success_rate']:.4f}, "
        f"succ_len_ratio_p50={v['success_only_len_ratio']['p50']:.3f}, "
        f"succ_loop={v['success_only_loop_rate']:.3f}"
    )
PY

# Step 4: Hero OD (B2 vs RNN vs Transformer)
run_step "VIZ_HERO_OD_B2_RNN_TR" \
  conda run -n dpl python -u tools/waycasd_plot_od_hero_figure.py \
    --phasec_json "$PHASEC_JSON" \
    --hero_label "BetaVAE64_FlowMu_K16_AL" \
    --method "BetaVAE64_FlowMu_K16_AL|greedy=${B2_PER}" \
    --method "RNN_AR_b10|beam=${RNN_PER}" \
    --method "Transformer_AR_b10|beam=${TR_PER}" \
    --way_features_npz "$WAY_FEATURES" \
    --city_grid_meta "0=${CITY_META}" \
    --coord_mode latlon \
    --use_basemap \
    --city 0 \
    --min_gt_routes 5 \
    --min_pred_success 3 \
    --min_self_diversity 0.3 \
    --hops_min 10 \
    --hops_max 80 \
    --k_pred_per_method 10 \
    --out_dir "${OUT_VIZ}/figA_hero"

# Step 5: Loop cases (Leaflet)
run_step "VIZ_LOOP_LEAFLET_B2" \
  conda run -n dpl python -u tools/waycasd_plot_loop_leaflet.py \
    --per_route_json "$B2_PER" \
    --way_features_npz "$WAY_FEATURES" \
    --city_grid_meta "0=${CITY_META}" \
    --out_html "${OUT_VIZ}/loop_cases_b2_k16_antiloop.html" \
    --mode greedy \
    --city 0 \
    --max_cases 10 \
    --only_failed \
    --sort_by loop_len

run_step "VIZ_LOOP_LEAFLET_TRANSFORMER" \
  conda run -n dpl python -u tools/waycasd_plot_loop_leaflet.py \
    --per_route_json "$TR_PER" \
    --way_features_npz "$WAY_FEATURES" \
    --city_grid_meta "0=${CITY_META}" \
    --out_html "${OUT_VIZ}/loop_cases_transformer_b10.html" \
    --mode beam \
    --city 0 \
    --max_cases 10 \
    --only_failed \
    --sort_by loop_len

# Step 6: len_ratio histogram (success-only)
run_step "VIZ_LEN_RATIO_HIST" \
  conda run -n dpl python -u - <<'PY'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cfg = [
    ("BetaVAE64_FlowMu_K16_AL", "_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B2_eval_k16_antiloop/per_route_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json", "greedy", "#d55e00"),
    ("RNN_AR_b10", "_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/baseline_eval/per_route_rnn_beam10_n5000.json", "beam", "#0072b2"),
    ("Transformer_AR_b10", "_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/baseline_eval/per_route_transformer_beam10_n5000.json", "beam", "#009e73"),
]

out_png = Path("_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/viz/len_ratio_success_only_hist.png")
out_pdf = Path("_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/viz/len_ratio_success_only_hist.pdf")
out_meta = Path("_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/viz/len_ratio_success_only_hist.meta.json")
out_png.parent.mkdir(parents=True, exist_ok=True)

meta = {}
plt.figure(figsize=(8.5, 5.2))
bins = np.linspace(1.0, 8.0, 36)
for name, path, dec, color in cfg:
    rows = json.loads(Path(path).read_text(encoding="utf-8")).get("per_route", [])
    vals = []
    for r in rows:
        d = r.get(dec, {})
        if bool(d.get("success", False)):
            x = float(d.get("len_ratio", float("nan")))
            if np.isfinite(x):
                vals.append(x)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size > 0:
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        lbl = f"{name} (p50={p50:.2f}, p95={p95:.2f}, n={arr.size})"
        plt.hist(arr, bins=bins, alpha=0.35, color=color, density=True, label=lbl)
        meta[name] = {"n": int(arr.size), "p50": p50, "p95": p95, "mean": float(arr.mean())}
    else:
        meta[name] = {"n": 0, "p50": float("nan"), "p95": float("nan"), "mean": float("nan")}

plt.xlabel("Length Ratio (pred_len / gt_len), success-only")
plt.ylabel("Density")
plt.title("Success-Only Length Ratio Distribution (Porto, n=5000)")
plt.xlim(1.0, 8.0)
plt.grid(alpha=0.2)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(out_png, dpi=180)
plt.savefig(out_pdf)
plt.close()

out_meta.write_text(json.dumps({"ok": True, "task": "len_ratio_hist_success_only", "methods": meta}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] saved: {out_png}")
print(f"[OK] saved: {out_pdf}")
print(f"[OK] saved: {out_meta}")
PY

echo ""
echo "======================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "DONE: Porto baseline eval + visualization 全部完成"
else
  echo "DONE with failures. FAILED STEPS:"
  for x in "${FAILED[@]}"; do
    echo "  - $x"
  done
fi
echo "OUT_ROOT: ${OUT_ROOT}"
echo "======================================================================"

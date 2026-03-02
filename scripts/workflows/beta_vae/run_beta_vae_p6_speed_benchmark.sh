#!/usr/bin/env bash

# P6: 训练/推理速度对比（CascadeTraj vs RNN-AR vs Transformer-AR）
# 说明：
# - 不重训，仅评估计时
# - 默认 n_routes=5000（全量口径）
# - 生成 speed_benchmark_summary.json

set -u

echo ">>> [Init] 进入仓库根目录"
PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJ_ROOT" || true

DATA_BASE="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${DATA_BASE}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${DATA_BASE}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_BASE}/W3_way_features/way_features.npz"
WAY_REGIONS="${DATA_BASE}/region_sweep/way_regions_louvain_res5_seed0.npz"
SPLIT_JSON="${DATA_BASE}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"
CITY_META="/home/jinlin/data/geoexplicit_data/porto_taxi/semantic/osm_road_prob_meta.json"

AE_CKPT="_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/A1_beta_vae_ae/ckpt_best.pt"
FLOW_CKPT="_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/A2_flow_on_mu/ckpt_best.pt"
RNN_CKPT="_sync/wsa/pi_verify/20260210_porto_phase1_s0/B2_rnn_ar_dev10p/ckpt_best.pt"
TR_CKPT="_sync/wsa/pi_verify/20260210_porto_phase1_s0/B3_transformer_ar_dev10p/ckpt_best.pt"

N_ROUTES=5000
OUT_ROOT="_sync/wsa/pi_verify/20260302_porto_speed_benchmark_s0"
OUT_CASCADE="${OUT_ROOT}/cascade_k16_al"
OUT_RNN="${OUT_ROOT}/rnn_b10"
OUT_TR="${OUT_ROOT}/transformer_b10"
mkdir -p "${OUT_CASCADE}" "${OUT_RNN}" "${OUT_TR}"

TIMING_CSV="${OUT_ROOT}/timing_raw.csv"
echo "name,seconds,rc,n_routes" > "${TIMING_CSV}"

FAILED_STEPS=()

run_timed_step() {
  local name="$1"
  local log_file="$2"
  shift 2
  echo ""
  echo "======================================================================"
  echo ">>> [${name}] START"
  echo ">>> Log: ${log_file}"
  echo "======================================================================"
  local t0 t1 sec rc
  t0=$(date +%s)
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${log_file}"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  sec=$((t1 - t0))
  echo "${name},${sec},${rc},${N_ROUTES}" >> "${TIMING_CSV}"
  if [ "${rc}" -ne 0 ]; then
    echo ">>> [${name}] FAILED (rc=${rc}, ${sec}s)"
    FAILED_STEPS+=("${name}")
  else
    echo ">>> [${name}] OK (${sec}s)"
  fi
}

echo ">>> [Preflight] 检查关键输入"
for f in "${WAY_ROUTES}" "${WAY_GRAPH}" "${WAY_FEATURES}" "${WAY_REGIONS}" "${SPLIT_JSON}" "${CITY_META}" "${AE_CKPT}" "${FLOW_CKPT}" "${RNN_CKPT}" "${TR_CKPT}"; do
  if [ -f "${f}" ]; then
    ls -lh "${f}"
  else
    echo "[MISS] ${f}"
  fi
done

run_timed_step "CascadeTraj_K16_AL" "${OUT_CASCADE}/run_eval.log" \
  conda run -n dpl python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_CKPT}" \
    --flow_ckpt "${FLOW_CKPT}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --latent_source flow \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --n_routes "${N_ROUTES}" \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --n_samples_per_route 16 \
    --sample_select dest_efficient \
    --decode_max_candidates 0 \
    --decode_candidate_policy first \
    --anti_loop_k 4 \
    --anti_loop_penalty 2.0 \
    --anti_loop_penalty_k 4 \
    --no_compare_beam \
    --city_grid_meta "0=${CITY_META}" \
    --eval_batch_size 256 \
    --out_json "${OUT_CASCADE}/binned.json" \
    --out_per_route_json "${OUT_CASCADE}/per_route.json" \
    --device cuda \
    --seed 0

run_timed_step "RNN_AR_B10" "${OUT_RNN}/run_eval.log" \
  conda run -n dpl python -u -m src.evaluation.unified_binned_eval \
    --method rnn_ar \
    --ckpt "${RNN_CKPT}" \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --city_grid_meta "0=${CITY_META}" \
    --n_routes "${N_ROUTES}" \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --beam_size 10 \
    --decode_max_candidates -1 \
    --eval_batch_size 256 \
    --out_json "${OUT_RNN}/binned.json" \
    --out_per_route_json "${OUT_RNN}/per_route.json" \
    --device cuda \
    --seed 0

run_timed_step "Transformer_AR_B10" "${OUT_TR}/run_eval.log" \
  conda run -n dpl python -u -m src.evaluation.unified_binned_eval \
    --method transformer_ar \
    --ckpt "${TR_CKPT}" \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --city_grid_meta "0=${CITY_META}" \
    --n_routes "${N_ROUTES}" \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --beam_size 10 \
    --decode_max_candidates -1 \
    --eval_batch_size 256 \
    --out_json "${OUT_TR}/binned.json" \
    --out_per_route_json "${OUT_TR}/per_route.json" \
    --device cuda \
    --seed 0

echo ""
echo ">>> [Summary] 生成速度对比汇总"
python - <<'PY'
import csv
import json
from pathlib import Path
import numpy as np

root = Path("_sync/wsa/pi_verify/20260302_porto_speed_benchmark_s0")
timing_csv = root / "timing_raw.csv"

name_to_binned = {
    "CascadeTraj_K16_AL": root / "cascade_k16_al" / "binned.json",
    "RNN_AR_B10": root / "rnn_b10" / "binned.json",
    "Transformer_AR_B10": root / "transformer_b10" / "binned.json",
}

timing = {}
with timing_csv.open("r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        name = row["name"]
        timing[name] = {
            "seconds": float(row["seconds"]),
            "rc": int(row["rc"]),
            "n_routes": int(row["n_routes"]),
        }

rows = []
for name, bp in name_to_binned.items():
    t = timing.get(name, None)
    if t is None or (not bp.exists()):
        rows.append({"name": name, "ok": False, "timing": t, "binned_json": str(bp)})
        continue
    b = json.loads(bp.read_text(encoding="utf-8")).get("global", {})
    sec = float(t["seconds"])
    n = int(t["n_routes"])
    rows.append(
        {
            "name": name,
            "ok": True,
            "seconds": sec,
            "routes_per_sec": float(n / sec) if sec > 0 else float("nan"),
            "success": float(b.get("success_rate", np.nan)),
            "hit_wall": float(b.get("hit_wall_rate", np.nan)),
            "loop": float(b.get("loop_rate", np.nan)),
            "len_ratio_mean": float(b.get("len_ratio_mean", np.nan)),
            "binned_json": str(bp),
        }
    )

out = {"ok": True, "task": "speed_benchmark", "rows": rows}
op = root / "speed_benchmark_summary.json"
op.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] saved: {op}")
print("name | sec | routes/s | success | hit_wall | loop | len_ratio_mean")
print("-" * 92)
for x in rows:
    if not x.get("ok", False):
        print(f"{x['name']} | MISSING")
        continue
    print(
        f"{x['name']} | {x['seconds']:.1f} | {x['routes_per_sec']:.3f} | {x['success']:.4f} | "
        f"{x['hit_wall']:.4f} | {x['loop']:.4f} | {x['len_ratio_mean']:.4f}"
    )
PY

echo ""
if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
  echo ">>> DONE with failures: ${FAILED_STEPS[*]}"
else
  echo ">>> DONE: 速度对比评估完成"
fi


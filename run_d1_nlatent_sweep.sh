#!/usr/bin/env bash
# ============================================================================
# D1: n_latent capacity sweep — 验证 latent 容量与 Flow 信息预算对齐
# ============================================================================
#
# 核心假设：
#   n_latent=64 → z 容量 16K dims，encoder 可编码 step-level 路径细节
#   Flow 从 (O,D) 只能重建宏观方向 → z_flow 远弱于 z_enc
#   E2 decoder 理性地学会忽略 z → 推理时 z 失效 + prefix 有误差 → 绕路循环
#
#   降低 n_latent → 强制 encoder 只编码宏观信息 → z_enc 与 z_flow 信息差缩小
#   → E2 decoder 无法/不需要忽略 z → 推理质量提升
#
# 实验设计：
#   n_latent ∈ {4, 8, 16} 三组，全部走完整管线：AE → Flow → E2 → 诊断 → 评估
#   对照组：baseline n_latent=64（已有，不重跑）
#
# 共享路径（不变）：
#   数据、way_graph、way_features、split_json、way_regions 全部复用 baseline
#
# 时间预估（4060 单卡）：
#   每组：AE ~90min + Flow ~20min + E2 ~50min + eval ~15min ≈ 3h
#   三组串行 ≈ 9h，并行两组（GPU 内存允许的话）≈ 6h
#   建议优先跑 n_latent=8，完成后根据结果决定 4/16
#
# ============================================================================

set -euo pipefail

# ---- Shared Paths ----
DATA_ROOT="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${DATA_ROOT}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${DATA_ROOT}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_ROOT}/W3_way_features/way_features.npz"
SPLIT_JSON="${DATA_ROOT}/W5_way_routes_strict_gate/od_split_min3_max160_seed0_dev10p.json"
WAY_REGIONS="${DATA_ROOT}/region_sweep/way_regions_louvain_res5_seed0.npz"
REGION_SEQ="${DATA_ROOT}/region_seq_res5/region_seq_min3_max160.npz"
CITY_GRID_META="${DATA_ROOT}/../../../porto_taxi/semantic/osm_road_prob_meta.json"
# NOTE: city_grid_meta path may differ on your machine; above is inferred from previous eval reports.
# If different, adjust accordingly.

SEED=0
DATE_TAG="20260219"

# ---- Sweep Values ----
# 优先跑 8；4 和 16 根据结果后续决定
N_LATENT_VALUES=(8)
# 如需全部：N_LATENT_VALUES=(4 8 16)

for NL in "${N_LATENT_VALUES[@]}"; do

  echo ""
  echo "================================================================"
  echo " n_latent = ${NL} — Full pipeline"
  echo "================================================================"

  EXP_DIR="_sync/wsa/pi_verify/${DATE_TAG}_porto_d1_nlatent${NL}_s${SEED}"
  mkdir -p "${EXP_DIR}"

  # ============================================================
  # Stage 1: AE Training (n_latent=${NL})
  # ============================================================
  # 与 baseline AE 完全相同的架构（cand_query, past_k=16），仅改 n_latent
  # baseline AE: n_epochs=100, lr=2e-4, batch_size=128, early_stop_patience=12
  # val_loss baseline(n64) = 0.1612, val_acc = 0.932
  # ============================================================
  AE_DIR="${EXP_DIR}/D1_ae_nL${NL}"
  AE_CKPT="${AE_DIR}/ckpt_best.pt"
  mkdir -p "${AE_DIR}"

  echo "[Stage 1] AE training — n_latent=${NL}"
  PYTHONUNBUFFERED=1 python -u -m src.training.train_way_casd_autoencoder \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --out_dir "${AE_DIR}" \
    --split_json "${SPLIT_JSON}" \
    --batch_size 128 \
    --n_epochs 100 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --seed ${SEED} \
    --min_hops 5 \
    --max_way_len 160 \
    --max_len 160 \
    --d_model 256 \
    --n_latent ${NL} \
    --n_heads 8 \
    --dropout 0.1 \
    --decoder_use_dest_dist \
    --decoder_use_cross_attn \
    --decoder_use_cand_query \
    --decoder_use_past_context \
    --decoder_past_k 16 \
    --decoder_past_n_layers 2 \
    --decoder_past_n_heads 4 \
    --early_stop_patience 12 \
    --save_every 5 \
    2>&1 | tee "${AE_DIR}/run_train.log"

  echo ""
  echo "[Stage 1 check] AE val_loss (from report.json):"
  python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('${AE_DIR}/report.json').read_text())
vl = r['best_val_loss']; ep = r['best_epoch']
print(f'  best_val_loss={vl:.4f}  best_epoch={ep}')
# Go/No-go: val_loss < 0.60 (n64 baseline=0.16, 允许 3-4x 退化)
if vl > 0.60:
    print('  ⚠ WARNING: val_loss > 0.60, AE may be too weak. Consider increasing n_latent.')
else:
    print('  ✓ Go: AE loss acceptable.')
"

  # ============================================================
  # Stage 1b: AE zenc_informativeness (验证 AE 本身 z 利用率)
  # ============================================================
  AE_ZENC_DIR="${EXP_DIR}/D1b_ae_zenc"
  mkdir -p "${AE_ZENC_DIR}"

  echo "[Stage 1b] AE zenc_informativeness — n_latent=${NL}"
  PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_zenc_informativeness \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_CKPT}" \
    --out_json "${AE_ZENC_DIR}/zenc_info_ae_nL${NL}_n5000.json" \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --n_routes 5000 \
    --seed ${SEED} \
    2>&1 | tee "${AE_ZENC_DIR}/run_zenc.log"

  echo "[Stage 1b check] AE T-S gap:"
  python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('${AE_ZENC_DIR}/zenc_info_ae_nL${NL}_n5000.json').read_text())
s = r['summary']
ts = s['true']['success_rate']; sh = s['shuffle']['success_rate']; ze = s['zero']['success_rate']
gap = ts - sh
print(f'  true={ts:.4f}  shuffle={sh:.4f}  zero={ze:.4f}  T-S gap={gap:.4f}')
# 关键判据：T-S gap > 0.30 说明 decoder 在利用 z
# 但 n_latent 更小时，AE true SR 也会更低，所以 gap 相对值更重要
if gap < 0.05:
    print('  ⚠ WARNING: T-S gap < 5pp, z may be useless.')
elif gap < 0.15:
    print('  ℹ T-S gap moderate. z carries some info.')
else:
    print('  ✓ Strong T-S gap. z is well-utilized by AE decoder.')
"

  # ============================================================
  # Stage 2: Flow Training (条件在新 AE 的 z 分布上)
  # ============================================================
  # baseline Flow: n_layers=4, noise_sigma=0.2, 60 epochs, cond_inject=xattn, use_region_seq
  # val_loss baseline(n64) = 0.2033
  # ============================================================
  FLOW_DIR="${EXP_DIR}/D2_flow_nL${NL}"
  FLOW_CKPT="${FLOW_DIR}/ckpt_best.pt"
  mkdir -p "${FLOW_DIR}"

  echo "[Stage 2] Flow training — n_latent=${NL}"
  PYTHONUNBUFFERED=1 python -u -m src.training.train_way_casd_flow \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_CKPT}" \
    --out_dir "${FLOW_DIR}" \
    --split_json "${SPLIT_JSON}" \
    --batch_size 128 \
    --n_epochs 60 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --seed ${SEED} \
    --min_hops 5 \
    --max_way_len 160 \
    --d_model 256 \
    --n_latent ${NL} \
    --n_layers 4 \
    --n_heads 8 \
    --dropout 0.1 \
    --noise_sigma 0.2 \
    --solver_steps 20 \
    --cond_inject xattn \
    --use_region_seq \
    --region_seq_npz "${REGION_SEQ}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --save_every 5 \
    2>&1 | tee "${FLOW_DIR}/run_train.log"

  echo "[Stage 2 check] Flow val_loss:"
  python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('${FLOW_DIR}/report.json').read_text())
vl = r['best_val_loss']; ep = r['best_epoch']
print(f'  best_val_loss={vl:.4f}  best_epoch={ep}')
# n_latent 越小，Flow 建模越容易，val_loss 应该更低
print('  (baseline n64 Flow val_loss=0.2033; smaller n_latent should be ≤ this)')
"

  # ============================================================
  # Stage 3: E2 Joint Fine-tune (decoder on Flow z)
  # ============================================================
  # baseline E2: batch_size=128, lr=1e-5, 100 epochs (40+60 cont)
  #              val_loss ≈ 0.925
  # 这里用 30 epochs 做快速验证，如果效果好再加训
  # ============================================================
  E2_DIR="${EXP_DIR}/D3_e2_nL${NL}"
  E2_CKPT="${E2_DIR}/ckpt_best.pt"
  mkdir -p "${E2_DIR}"

  echo "[Stage 3] E2 fine-tune — n_latent=${NL}"
  PYTHONUNBUFFERED=1 python -u -m src.training.train_way_casd_decoder_joint \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${AE_CKPT}" \
    --flow_ckpt "${FLOW_CKPT}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --out_dir "${E2_DIR}" \
    --split_json "${SPLIT_JSON}" \
    --batch_size 128 \
    --n_epochs 30 \
    --lr 1e-5 \
    --weight_decay 0.0 \
    --seed ${SEED} \
    --min_hops 5 \
    --max_way_len 160 \
    --early_stop_patience 8 \
    --max_grad_norm 1.0 \
    --save_every 1 \
    --log_every_batches 200 \
    2>&1 | tee "${E2_DIR}/run_train.log"

  echo "[Stage 3 check] E2 val_loss:"
  python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('${E2_DIR}/report.json').read_text())
vl = r['best_val_loss']; ep = r['best_epoch']
print(f'  best_val_loss={vl:.4f}  best_epoch={ep}')
"

  # ============================================================
  # Stage 4a: E2 zenc_informativeness（关键诊断：E2 后 z 利用率）
  # ============================================================
  E2_ZENC_DIR="${EXP_DIR}/D4a_e2_zenc"
  mkdir -p "${E2_ZENC_DIR}"

  echo "[Stage 4a] E2 zenc_informativeness — n_latent=${NL}"
  PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_zenc_informativeness \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${E2_CKPT}" \
    --out_json "${E2_ZENC_DIR}/zenc_info_e2_nL${NL}_n5000.json" \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --n_routes 5000 \
    --seed ${SEED} \
    2>&1 | tee "${E2_ZENC_DIR}/run_zenc.log"

  echo "[Stage 4a check] E2 T-S gap:"
  python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('${E2_ZENC_DIR}/zenc_info_e2_nL${NL}_n5000.json').read_text())
s = r['summary']
ts = s['true']['success_rate']; sh = s['shuffle']['success_rate']; ze = s['zero']['success_rate']
gap = ts - sh
print(f'  true={ts:.4f}  shuffle={sh:.4f}  zero={ze:.4f}  T-S gap={gap:.4f}')
print()
# 关键判据：与 baseline E2 (T-S gap=0.0994) 比较
# 如果 gap 显著 > 0.10，说明 n_latent 缩小确实保留了 z 利用率
if gap > 0.20:
    print('  ✓✓ STRONG IMPROVEMENT: E2 T-S gap > 20pp (baseline=9.94pp)')
elif gap > 0.10:
    print('  ✓ Improvement over baseline E2 T-S gap (9.94pp)')
else:
    print('  ✗ No improvement in E2 T-S gap. Hypothesis may be wrong.')
"

  # ============================================================
  # Stage 4b: Binned Eval — K1 greedy + beam (快速)
  # ============================================================
  EVAL_DIR="${EXP_DIR}/D4b_eval"
  mkdir -p "${EVAL_DIR}"

  echo "[Stage 4b] Binned eval K1 greedy+beam — n_latent=${NL}"
  PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${E2_CKPT}" \
    --flow_ckpt "${FLOW_CKPT}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --out_json "${EVAL_DIR}/binned_e2_nL${NL}_k1_beam10_n2000.json" \
    --latent_source flow \
    --n_samples_per_route 1 \
    --sample_select first \
    --n_routes 2000 \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --beam_size 10 \
    --compare_beam \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --city_grid_meta "0=${CITY_GRID_META}" \
    --seed ${SEED} \
    2>&1 | tee "${EVAL_DIR}/run_eval_k1.log"

  # ============================================================
  # Stage 4c: Binned Eval — K8 dest select
  # ============================================================
  echo "[Stage 4c] Binned eval K8 dest — n_latent=${NL}"
  PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${E2_CKPT}" \
    --flow_ckpt "${FLOW_CKPT}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --out_json "${EVAL_DIR}/binned_e2_nL${NL}_k8_dest_n2000.json" \
    --latent_source flow \
    --n_samples_per_route 8 \
    --sample_select dest \
    --n_routes 2000 \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --beam_size 10 \
    --no_compare_beam \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --city_grid_meta "0=${CITY_GRID_META}" \
    --seed ${SEED} \
    2>&1 | tee "${EVAL_DIR}/run_eval_k8dest.log"

  # ============================================================
  # Stage 4d: Binned Eval — K16 dest select (与 baseline 可比)
  # ============================================================
  echo "[Stage 4d] Binned eval K16 dest — n_latent=${NL}"
  PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "${WAY_ROUTES}" \
    --way_graph_npz "${WAY_GRAPH}" \
    --way_features_npz "${WAY_FEATURES}" \
    --ae_ckpt "${E2_CKPT}" \
    --flow_ckpt "${FLOW_CKPT}" \
    --way_regions_npz "${WAY_REGIONS}" \
    --out_json "${EVAL_DIR}/binned_e2_nL${NL}_k16_dest_n5000.json" \
    --latent_source flow \
    --n_samples_per_route 16 \
    --sample_select dest \
    --n_routes 5000 \
    --min_hops 5 \
    --max_way_len 160 \
    --max_decode_len 160 \
    --beam_size 10 \
    --no_compare_beam \
    --split_json "${SPLIT_JSON}" \
    --split_part test \
    --city_grid_meta "0=${CITY_GRID_META}" \
    --seed ${SEED} \
    2>&1 | tee "${EVAL_DIR}/run_eval_k16dest.log"

  # ============================================================
  # Stage 5: Summary Report
  # ============================================================
  echo ""
  echo "================================================================"
  echo " SUMMARY: n_latent=${NL}"
  echo "================================================================"
  python3 -c "
import json, pathlib

exp = '${EXP_DIR}'

# AE
ae_r = json.loads(pathlib.Path(f'{exp}/D1_ae_nL${NL}/report.json').read_text())
print(f'AE:  val_loss={ae_r[\"best_val_loss\"]:.4f}  best_epoch={ae_r[\"best_epoch\"]}')

# AE zenc
ae_z = json.loads(pathlib.Path(f'{exp}/D1b_ae_zenc/zenc_info_ae_nL${NL}_n5000.json').read_text())['summary']
ae_gap = ae_z['true']['success_rate'] - ae_z['shuffle']['success_rate']
print(f'AE zenc: true={ae_z[\"true\"][\"success_rate\"]:.4f} shuf={ae_z[\"shuffle\"][\"success_rate\"]:.4f} gap={ae_gap:.4f}')

# Flow
fl_r = json.loads(pathlib.Path(f'{exp}/D2_flow_nL${NL}/report.json').read_text())
print(f'Flow: val_loss={fl_r[\"best_val_loss\"]:.4f}  best_epoch={fl_r[\"best_epoch\"]}')

# E2
e2_r = json.loads(pathlib.Path(f'{exp}/D3_e2_nL${NL}/report.json').read_text())
print(f'E2:  val_loss={e2_r[\"best_val_loss\"]:.4f}  best_epoch={e2_r[\"best_epoch\"]}')

# E2 zenc
e2_z = json.loads(pathlib.Path(f'{exp}/D4a_e2_zenc/zenc_info_e2_nL${NL}_n5000.json').read_text())['summary']
e2_gap = e2_z['true']['success_rate'] - e2_z['shuffle']['success_rate']
print(f'E2 zenc: true={e2_z[\"true\"][\"success_rate\"]:.4f} shuf={e2_z[\"shuffle\"][\"success_rate\"]:.4f} gap={e2_gap:.4f}')

# Eval K8 dest
k8 = json.loads(pathlib.Path(f'{exp}/D4b_eval/binned_e2_nL${NL}_k8_dest_n2000.json').read_text())
cells = k8['overall']['greedy']['cells']
total_n = sum(c['n'] for c in cells.values())
total_succ = sum(c['n'] * c['success_rate'] for c in cells.values())
k8_sr = total_succ / total_n if total_n > 0 else 0
k8_lr = sum(c['n'] * c['len_ratio']['mean'] for c in cells.values()) / total_n if total_n > 0 else 0
k8_loop = sum(c['n'] * c['loop_rate'] for c in cells.values()) / total_n if total_n > 0 else 0
print(f'K8 dest:  SR={k8_sr:.4f}  len_ratio={k8_lr:.2f}  loop={k8_loop:.3f}')

# Eval K16 dest
k16 = json.loads(pathlib.Path(f'{exp}/D4b_eval/binned_e2_nL${NL}_k16_dest_n5000.json').read_text())
cells16 = k16['overall']['greedy']['cells']
total_n16 = sum(c['n'] for c in cells16.values())
total_succ16 = sum(c['n'] * c['success_rate'] for c in cells16.values())
k16_sr = total_succ16 / total_n16 if total_n16 > 0 else 0
k16_lr = sum(c['n'] * c['len_ratio']['mean'] for c in cells16.values()) / total_n16 if total_n16 > 0 else 0
k16_loop = sum(c['n'] * c['loop_rate'] for c in cells16.values()) / total_n16 if total_n16 > 0 else 0
print(f'K16 dest: SR={k16_sr:.4f}  len_ratio={k16_lr:.2f}  loop={k16_loop:.3f}')

print()
print('=== COMPARISON vs BASELINE (n_latent=64) ===')
print('Baseline AE: val_loss=0.1612  zenc T-S gap=+62.28pp')
print('Baseline E2: val_loss=0.9252  zenc T-S gap=+9.94pp  K16 SR≈0.65  len_ratio≈5.4')
print(f'This (nL{${NL}}): E2 T-S gap={e2_gap:.4f}  K16 SR={k16_sr:.4f}  len_ratio={k16_lr:.2f}')
if e2_gap > 0.20 and k16_sr > 0.55:
    print('>>> STRONG GO: Proceed with full ablation sweep (nL=4,8,16,32,64)')
elif e2_gap > 0.10 or k16_sr > 0.50:
    print('>>> PARTIAL: Some improvement, consider adjusting n_latent or E2 epochs')
else:
    print('>>> NO-GO: Hypothesis not confirmed at nL=${NL}')
"

  echo ""
  echo "All artifacts saved to: ${EXP_DIR}"
  echo ""

done

echo "================================================================"
echo "D1 n_latent sweep complete."
echo "================================================================"

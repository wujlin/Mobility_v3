# Way-CASD 实验 Checklist

> 目标：缩小 Oracle（AE oracle，上界）→ Flow（端到端生成） 的性能差距
> 更新日期：2026-02-06

---

## 当前最佳配置

```yaml
AE/Decoder: E2 joint fine-tune (cont e60), ckpt_best.pt
Flow: W11 RegionSeq xattn (n_layers=6)
Decode: beam=10, softP=2.0 K=4, maxcand=0
Region: AR constraint, relaxed, dest_region fallback
```

### 当前性能（E2 cont e60）

| Bin | Baseline | E2续训 | Oracle | 剩余Gap |
|-----|----------|--------|--------|---------|
| [5,10) | 62.5% | 91.7% | 100% | 8.3pp |
| [10,20) | 67.0% | 86.6% | 96.9% | 10.3pp |
| [20,30) | 56.3% | 69.0% | 91.5% | **22.5pp** |
| [30,40) | 57.5% | 80.8% | 91.8% | 11.0pp |
| [40,60) | 32.5% | 51.9% | 85.7% | **33.8pp** ← 最大瓶颈 |
| [60,+) | 44.1% | 64.7% | 85.3% | 20.6pp |
| **Overall** | 54.25% | **74.5%** | ~92% | ~17pp |

### 关键诊断数据
- E1 latent diagnosis: `cos(z_flow, z_gt) ≈ 0.31`（严重misalign）
- D4 空间审计: hit_wall终点 outdeg p50=2（选错corridor，非局部confusion）
- [40,60) 失败模式: hit_wall=46.8%, loop=37.7%（几乎全是hit_wall）

---

## 待执行实验（按优先级排序）

### P0：OD-Disjoint Best-of-K（论文主表关键；零训练成本）

> 说明：PI 要求的 EXP-1~EXP-4 本质上是同一组 eval，只是改 `n_samples_per_route` / `sample_select` 并强制输出 per_route。
> 我们把它整理成 3 个可直接跑的实验（best-of-8、K-sweep、oracle 上界），并把 per_route dump 作为默认输出。

#### EXP-1：OD-Disjoint best-of-8（deployable）
- 只改两项：`--n_samples_per_route 8 --sample_select dest`
- 输出：`_sync/wsa/pi_verify/20260206_od_disjoint_s0/p0_bestofk/`

#### EXP-2：K sweep（K=1/2/4/8/16）
- 目的：画 diversity 曲线（success vs K），并报告 any-success/mean-sample-success（来自 per_route）
- 输出同上（每个 K 一份 binned +（可选）per_route）

#### EXP-3：Oracle decode 上界（标定 flow 损失）
- `--latent_source gt`（GT→AE.encode→Decoder）
- 输出同上（binned + per_route）

#### EXP-4：paired 分析（McNemar + shape paired diff）
- 依赖 EXP-1/3 的 `per_route_*.json`
- 输出：`paired_*.md`（直接放到同一目录，便于 git review）

---

### P0：执行命令（推荐直接复制粘贴；无 exit）

> 建议：先从 **已有 OD-disjoint K=1 评测** 自动抽取输入路径，避免手填出错。

```bash
cd ~/projects/Mobility_v3

export OD_ROOT="_sync/wsa/pi_verify/20260206_od_disjoint_s0"
export BASE_JSON="$OD_ROOT/eval/binned_waycasd_flow_test_n200pc.json"
export OUT_DIR="$OD_ROOT/p0_bestofk"
mkdir -p "$OUT_DIR"

# 0) 生成一份可 source 的环境变量脚本（从 BASE_JSON 自动读 inputs）
python - <<'PY'
import json, os, shlex
from pathlib import Path

base = Path(os.environ.get("BASE_JSON", "_sync/wsa/pi_verify/20260206_od_disjoint_s0/eval/binned_waycasd_flow_test_n200pc.json"))
obj = json.loads(base.read_text(encoding="utf-8"))
inp = obj["inputs"]

items = {
  "WAY_ROUTES_NPZ": inp["way_routes_npz"],
  "WAY_GRAPH_NPZ": inp["way_graph_npz"],
  "WAY_FEATS_NPZ": inp["way_features_npz"],
  "AE_CKPT": inp["ae_ckpt"],
  "FLOW_CKPT": inp["flow_ckpt"],
  "WAY_REGIONS_NPZ": inp["way_regions_npz"],
  "REGION_AR_CKPT": inp["region_ar_ckpt"],
  "SPLIT_JSON": inp["split_json"],
  "SPLIT_PART": inp["split_part"],
  "DET_META": inp["city_grid_meta"]["0"],
  "COL_META": inp["city_grid_meta"]["1"],
}
out = Path(os.environ.get("OUT_DIR", "_sync/wsa/pi_verify/20260206_od_disjoint_s0/p0_bestofk")) / "vars.sh"
lines = ["#!/usr/bin/env bash"]
for k, v in items.items():
  lines.append(f"export {k}={shlex.quote(str(v))}")
out.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(f"[OK] wrote: {out}")
PY

source "$OUT_DIR/vars.sh"

echo ">>> Preflight..."
ls -lh "$WAY_ROUTES_NPZ" "$WAY_GRAPH_NPZ" "$WAY_FEATS_NPZ" "$AE_CKPT" "$FLOW_CKPT" "$WAY_REGIONS_NPZ" "$REGION_AR_CKPT" "$DET_META" "$COL_META" "$SPLIT_JSON"
```

**EXP-1：best-of-8（deployable dest selection）**
```bash
PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" \
  --city_grid_meta "1=$COL_META" \
  --split_json "$SPLIT_JSON" --split_part "$SPLIT_PART" \
  --ae_ckpt "$AE_CKPT" \
  --latent_source flow --flow_ckpt "$FLOW_CKPT" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --region_constraint ar --region_ar_ckpt "$REGION_AR_CKPT" \
  --region_constraint_mode relaxed --region_constraint_fallback dest_region \
  --n_routes 200 --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --decode_candidate_policy first --decode_max_candidates 0 \
  --beam_size 10 \
  --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
  --n_samples_per_route 8 --sample_select dest \
  --out_json "$OUT_DIR/binned_flow_bestof8_dest.json" \
  --out_per_route_json "$OUT_DIR/per_route_flow_bestof8_dest.json" \
  |& tee "$OUT_DIR/run_bestof8_dest.log"
```

**EXP-2：K sweep（dest selection）**
```bash
for K in 1 2 4 8 16; do
  OUT_JSON="$OUT_DIR/binned_flow_bestof${K}_dest.json"
  OUT_PR="$OUT_DIR/per_route_flow_bestof${K}_dest.json"
  EXTRA_PR=""
  # 只对 K>=8 dump per_route（K=1/2/4 只保留 binned，避免仓库膨胀）
  if [[ "$K" -ge 8 ]]; then EXTRA_PR="--out_per_route_json $OUT_PR"; fi

  PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "$WAY_ROUTES_NPZ" \
    --way_graph_npz "$WAY_GRAPH_NPZ" \
    --way_features_npz "$WAY_FEATS_NPZ" \
    --city_grid_meta "0=$DET_META" \
    --city_grid_meta "1=$COL_META" \
    --split_json "$SPLIT_JSON" --split_part "$SPLIT_PART" \
    --ae_ckpt "$AE_CKPT" \
    --latent_source flow --flow_ckpt "$FLOW_CKPT" \
    --way_regions_npz "$WAY_REGIONS_NPZ" \
    --region_constraint ar --region_ar_ckpt "$REGION_AR_CKPT" \
    --region_constraint_mode relaxed --region_constraint_fallback dest_region \
    --n_routes 200 --min_hops 5 --max_way_len 160 --max_decode_len 160 \
    --decode_candidate_policy first --decode_max_candidates 0 \
    --beam_size 10 \
    --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
    --n_samples_per_route "$K" --sample_select dest \
    --out_json "$OUT_JSON" \
    $EXTRA_PR \
    |& tee "$OUT_DIR/run_bestof${K}_dest.log"
done
```

**（可选）画曲线：success vs K（需要本仓库的 `tools/waycasd_plot_bestofk_curve.py`）**
```bash
PYTHONUNBUFFERED=1 python -u tools/waycasd_plot_bestofk_curve.py \
  --out_dir "$OUT_DIR/fig" \
  --title "OD-disjoint best-of-K (dest selection)" \
  --json "$OUT_DIR/binned_flow_bestof1_dest.json" \
  --json "$OUT_DIR/binned_flow_bestof2_dest.json" \
  --json "$OUT_DIR/binned_flow_bestof4_dest.json" \
  --json "$OUT_DIR/binned_flow_bestof8_dest.json" \
  --json "$OUT_DIR/binned_flow_bestof16_dest.json"
```

**EXP-3：Oracle decode 上界（GT latent，上界）**
```bash
PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" \
  --city_grid_meta "1=$COL_META" \
  --split_json "$SPLIT_JSON" --split_part "$SPLIT_PART" \
  --ae_ckpt "$AE_CKPT" \
  --latent_source gt \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --region_constraint ar --region_ar_ckpt "$REGION_AR_CKPT" \
  --region_constraint_mode relaxed --region_constraint_fallback dest_region \
  --n_routes 200 --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --decode_candidate_policy first --decode_max_candidates 0 \
  --beam_size 10 \
  --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
  --out_json "$OUT_DIR/binned_oracle_gtlatent.json" \
  --out_per_route_json "$OUT_DIR/per_route_oracle_gtlatent.json" \
  |& tee "$OUT_DIR/run_oracle_gtlatent.log"
```

**EXP-4：paired（McNemar + shape）**
```bash
# K=1 vs best-of-8（deployable）
PYTHONUNBUFFERED=1 python -u tools/waycasd_paired_compare.py \
  --a_json "$OD_ROOT/eval/per_route_waycasd_flow_test_n200pc.json" \
  --b_json "$OUT_DIR/per_route_flow_bestof8_dest.json" \
  --a_name "Flow(K=1)" --b_name "Flow(best-of-8,dest)" \
  --key beam \
  --out_md "$OUT_DIR/paired_k1_vs_bestof8_dest.md"

# Flow(best-of-8,dest) vs Oracle(gt latent)
PYTHONUNBUFFERED=1 python -u tools/waycasd_paired_compare.py \
  --a_json "$OUT_DIR/per_route_flow_bestof8_dest.json" \
  --b_json "$OUT_DIR/per_route_oracle_gtlatent.json" \
  --a_name "Flow(best-of-8,dest)" --b_name "Oracle(gt latent)" \
  --key beam \
  --out_md "$OUT_DIR/paired_bestof8_dest_vs_oracle.md"
```

---

### P1：A2 Flow CFG（需重训Flow）

> 本轮已被 PI 暂时降级：建议先把 OD-disjoint P0（EXP-1~4）跑完整，再决定是否重训。

**目标**：增强Flow对condition的利用

**方法**：
1. 重训Flow with `--cond_dropout_p 0.1`
2. 推理时用 `--flow_cfg_scale {1.5, 2.0, 3.0}` sweep

**预期收益**：+3-5pp（如果condition利用不充分）

**输出**：`_sync/wsa/pi_verify/A2_flow_cfg_s0/`

---

### P2：A1 Flow更深（可选，依赖P0/P1结果）

**条件**：仅当B3显示"diversity不够"时才执行

**方法**：`n_layers 6→8`（保持d_model=256）

**注意**：不改d_model，因为需要AE/latent同步升级

---

## 已放弃的实验方向

| 实验 | 放弃原因 |
|------|----------|
| A3: 多步refinement | 架构复杂度太高，收益不确定 |
| B1: Region AR更强 | D4审计显示hit_wall发生在低度节点，Region AR不是瓶颈 |
| B2: Corridor-level loss | 需要设计新loss，"corridor alignment"定义本身是难题 |
| C1: E2继续训练 | **已完成**。best_epoch=38/40，val_loss已收敛(0.1913) |
| C2: RL fine-tune | D2实验已证明简单reward无效，RL训练不稳定 |
| C3: Decoder ensemble | 推理成本K倍，B3已cover多样本思路 |

---

## Partner 执行指南

### P0: B3 Best-of-K 评测

```bash
# 环境变量（请根据实际路径设置）
export WAY_ROUTES_NPZ="..."
export WAY_GRAPH_NPZ="..."
export WAY_FEATS_NPZ="..."
export DET_META="/home/jinlin/data/geoexplicit_data/worldtrace/detroit_core_v1/osm_road_prob_meta.json"
export COL_META="/home/jinlin/data/geoexplicit_data/worldtrace/columbus_core_v1/osm_road_prob_meta.json"
export AE_CKPT="_sync/wsa/pi_verify/E2_joint_finetune_s0_cont_e60/ckpt_best.pt"
export FLOW_CKPT="..."  # W11 Flow checkpoint
export WAY_REGIONS_NPZ="..."
export REGION_AR_CKPT="..."
export OUT_DIR="_sync/wsa/pi_verify/B3_bestofK_s0"

mkdir -p "$OUT_DIR"

# 1) Deployable best-of-8（不看GT，可部署）
PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" \
  --city_grid_meta "1=$COL_META" \
  --ae_ckpt "$AE_CKPT" \
  --latent_source flow --flow_ckpt "$FLOW_CKPT" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --region_constraint ar --region_ar_ckpt "$REGION_AR_CKPT" \
  --region_constraint_mode relaxed --region_constraint_fallback dest_region \
  --n_routes 200 --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --decode_candidate_policy first --decode_max_candidates 0 \
  --beam_size 10 \
  --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
  --n_samples_per_route 8 --sample_select dest \
  --out_json "$OUT_DIR/binned_flow_bestof8_dest.json" \
  |& tee "$OUT_DIR/run_bestof8_dest.log"

# 2) Oracle best-of-8（上界，用GT选最优）
PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" \
  --city_grid_meta "1=$COL_META" \
  --ae_ckpt "$AE_CKPT" \
  --latent_source flow --flow_ckpt "$FLOW_CKPT" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --region_constraint ar --region_ar_ckpt "$REGION_AR_CKPT" \
  --region_constraint_mode relaxed --region_constraint_fallback dest_region \
  --n_routes 200 --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --decode_candidate_policy first --decode_max_candidates 0 \
  --beam_size 10 \
  --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
  --n_samples_per_route 8 --sample_select best \
  --out_json "$OUT_DIR/binned_flow_bestof8_oraclebest.json" \
  |& tee "$OUT_DIR/run_bestof8_oraclebest.log"
```

### P1: A2 Flow CFG 重训

```bash
export REGION_SEQ_NPZ="_sync/wsa/pi_verify/20260201_min5_candq1_past8_len160_s0/regions_louvain_res1p0_v1/region_seq_min5_max160.npz"

# 建议：Flow ckpt 放数据盘（大文件），评测/日志放 _sync（便于 git review）
export OUT_A2_FLOW="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/W12_train_flow_cfgdrop0p1_s0"
export OUT_A2_SYNC="_sync/wsa/pi_verify/A2_flow_cfg_s0"
mkdir -p "$OUT_A2_FLOW" "$OUT_A2_SYNC"

# 1) 重训Flow with condition dropout
PYTHONUNBUFFERED=1 python -u -m src.training.train_way_casd_flow \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$AE_CKPT" \
  --out_dir "$OUT_A2_FLOW" \
  --min_hops 5 --max_way_len 160 \
  --cond_inject xattn --use_region_seq \
  --region_seq_npz "$REGION_SEQ_NPZ" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --cond_dropout_p 0.1 \
  --batch_size 256 --num_workers 16 --n_epochs 60 \
  --device cuda --seed 0 \
  |& tee "$OUT_A2_SYNC/run_train.log"

# 2) CFG scale sweep（训练完成后）
for CFG in 1.5 2.0 3.0; do
  PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
    --way_routes_npz "$WAY_ROUTES_NPZ" \
    --way_graph_npz "$WAY_GRAPH_NPZ" \
    --way_features_npz "$WAY_FEATS_NPZ" \
    --city_grid_meta "0=$DET_META" \
    --city_grid_meta "1=$COL_META" \
    --ae_ckpt "$AE_CKPT" \
    --latent_source flow --flow_ckpt "$OUT_A2_FLOW/ckpt_best.pt" \
    --flow_cfg_scale $CFG \
    --way_regions_npz "$WAY_REGIONS_NPZ" \
    --region_constraint ar --region_ar_ckpt "$REGION_AR_CKPT" \
    --region_constraint_mode relaxed --region_constraint_fallback dest_region \
    --n_routes 200 --min_hops 5 --max_way_len 160 --max_decode_len 160 \
    --decode_candidate_policy first --decode_max_candidates 0 \
    --beam_size 10 \
    --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
    --out_json "$OUT_A2_SYNC/binned_flow_cfg${CFG}.json" \
    |& tee "$OUT_A2_SYNC/run_cfg${CFG}.log"
done
```

### P2: A1 Flow 更深（n_layers 6→8，保持 d_model=256）

```bash
# 仅当 B3 显示 z_flow diversity 不够时再跑（否则优先 A2）
export OUT_A1_FLOW="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/W12_train_flow_deeperL8_s0"
export OUT_A1_SYNC="_sync/wsa/pi_verify/A1_flow_deeperL8_s0"
mkdir -p "$OUT_A1_FLOW" "$OUT_A1_SYNC"

# 1) 训练更深的 Flow（不启用 CFG）
PYTHONUNBUFFERED=1 python -u -m src.training.train_way_casd_flow \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$AE_CKPT" \
  --out_dir "$OUT_A1_FLOW" \
  --min_hops 5 --max_way_len 160 \
  --d_model 256 --n_layers 8 --n_heads 8 \
  --cond_inject xattn --use_region_seq \
  --region_seq_npz "$REGION_SEQ_NPZ" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --cond_dropout_p 0.0 \
  --batch_size 256 --num_workers 16 --n_epochs 60 \
  --device cuda --seed 0 \
  |& tee "$OUT_A1_SYNC/run_train.log"

# 2) 评测
PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --city_grid_meta "0=$DET_META" \
  --city_grid_meta "1=$COL_META" \
  --ae_ckpt "$AE_CKPT" \
  --latent_source flow --flow_ckpt "$OUT_A1_FLOW/ckpt_best.pt" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --region_constraint ar --region_ar_ckpt "$REGION_AR_CKPT" \
  --region_constraint_mode relaxed --region_constraint_fallback dest_region \
  --n_routes 200 --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --decode_candidate_policy first --decode_max_candidates 0 \
  --beam_size 10 \
  --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
  --out_json "$OUT_A1_SYNC/binned_eval_flow_n200pc.json" \
  |& tee "$OUT_A1_SYNC/run_eval.log"
```

---

## 结果解读指南

### B3 结果解读
1. 对比 `binned_flow_bestof8_oraclebest.json` vs 当前E2 (74.5%)
   - 如果 oracle-best-of-8 > 85% → **z_flow diversity足够**，问题是"选不对"，后续考虑训练一个selector
   - 如果 oracle-best-of-8 ≈ 75-80% → **z_flow diversity不够**，需要改进Flow本身（做A2）
2. 对比 `dest` vs `best` 选择策略的差距
   - 差距大 → 可部署的selector有改进空间

### A2 结果解读
1. 对比不同CFG scale的success rate
2. 最优CFG scale应该在1.5-2.0之间（太大会过拟合condition）

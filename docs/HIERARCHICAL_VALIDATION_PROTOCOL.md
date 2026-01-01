# Hierarchical 路线验证协议（硬约束版，避免走偏）

> 目的：把 “trip-level 决策 + segment-level 执行” 的分层路线，变成**可执行、可止损、可归因**的工程流程。  
> 原则：不为赶时间做 trivial 设计；不做无意义烧卡；所有止损线必须可复现（固定统计口径 + CI + 噪声地板）。

> **阶段性结果解读**：见 `docs/PHASE_C_RESULTS.md`（Phase C：Macro Hard Support + AR + DetRes）。
> **下一阶段路线图**：见 `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`（OSM 道路先验（软） + 拓扑 + 城市语义 + AR + Diffusion 多模态）。

**重要说明（避免口径混淆）**
- 本协议记录的是 **Phase C：Hard Support + AR + DetRes** 的已验证基线与审计方法（用于锁定归因/止损）。
- Phase D 主线将把 OSM/拓扑/语义作为**软先验特征**进入模型，**不在训练环节使用 masked softmax/hard cut**；Hard Support 仅保留为诊断上界与审计工具，不作为“能力”宣称。
- 本协议中的路径与命令示例默认基于旧数据（深圳出租车 dt=30s）；当主数据切换到 WorldTrace×Detroit（1Hz matched）时，需要按 `docs/DATA_CONTRACT.md` 与 `docs/WORDTRACE_UNITRAJ.md` 更新数据路径与窗口构造口径。

---

## 主线结论（可证伪承诺）

- **分层扩散最可能解决的问题**：`p(τ|o,d)` 是多路线模态的混合分布，端到端 score 学到的是“责任加权平均梯度场”→ 低频拓扑模态（detour）被平均掉，表现为 **Destination Gravity / 直线坍缩 + 抖动**。  
- **分层扩散不保证解决的问题**：若“不物理”来自**条件信息缺失（缺地图/可行域/规则）**或 micro 执行本身物理建模不足，单靠分层并不能凭空补齐约束。

因此我们把分层写成一个**可证伪承诺**（止损线）：
- **只有当 Oracle(z) 条件下**（z=waypoints/anchors），micro 能在“不破坏 coarse 拓扑”的前提下显著改善物理纹理指标，分层才是主线；
- 否则更可能需要：weak-map/graph 或更强的 micro 物理约束/表示（而不是继续堆层级扩散）。

---

## 当前事实快照（避免重复烧卡）

> 这一节只记录“已跑出的硬事实”，用于同步进度与止损。

- 数据版本：`data/processed_passenger_dt30`（Passenger-only，dt=30s，strict sanity check PASS）。
- GT windows：`data/experiments/gt_passenger_dt30_test/samples.npz`（默认 N=20000, obs=8, pred=12）。
- Go/No-Go 1（GT，OD 多模态证据）：
  - `od_bin=12`：`weighted_multimodal_ratio ≈ 0.109`
  - `od_bin=16`：`weighted_multimodal_ratio ≈ 0.197`
- Go/No-Go 2（GT，Oracle waypoint 的 skeleton-only）：
  - `Waypoint Gate (rdp_dev, K=2, linear)`：`collision_rate_any ≈ 0.0884 < 0.10`（通过硬可行性门槛）
  - `detour_validity`：`RDPK2` 在 overall + detour subset 上显著优于 `StraightK0`（粗拓扑可由 2 个 waypoint 承载）
- Go/No-Go 3（Oracle waypoint 条件下的 micro executability，三对照，N=400）：
  - `skeleton-only`（prior=skeleton_wp）：
    - `plot_physical_stats`：`JSD_Turn=0.0717, JSD_Speed=0.1786, JSD_Accel=0.2748; DCV_speed=0%, DCV_accel=0%`
  - `deterministic residual (DetRes)`：**可学且显著改善分布一致性**（强证据）
    - `plot_physical_stats`：`JSD_Turn=0.0368, JSD_Speed=0.1282, JSD_Accel=0.1872; DCV_speed=1.1818%, DCV_accel=0%`
    - `detour_validity`：overall 的 `turn@{1,2,4,8}` 与 `max_dev_ratio` 更接近 GT；detour 子集也不劣于 skeleton-only（粗拓扑不被破坏）
  - `diffusion residual (DiffRes)`：**灾难性失败（止损）**
    - `plot_physical_stats`：`JSD_Turn=0.4552, JSD_Speed=0.3189, JSD_Accel=0.7207; DCV_speed=24.5386%, DCV_accel=73.3825%`
    - `detour_validity`：overall 的 `turn@{1,2,4,8}` 与 `len_ratio/max_dev_ratio` 显著更差（micro 不可执行）
- 决策（KISS + 可归因）：**micro 执行层先固定为 DetRes**，DiffRes micro 路线进入止损分支（不再烧卡），主线进入 Phase 3（学习 macro waypoint/anchor）。

- Phase 3（Macro z-diffusion）当前阻塞点（detour-hard，N=400，K=20）：
  - `MacroSkel`（无过滤，原生采样）：`collision_rate_any ≈ 0.3049`，`waypoint_offroad_rate ≈ 0.1765` → **G1 不通过**（模型尚未学会可行域约束）。
  - `MacroSkel + FeasibleGate`（采样端 accept/reject，oversample=3）：`collision_rate_any ≈ 0.03`，`waypoint_offroad_rate ≈ 0.0591` → **仅说明分布中存在足够可行质量可被筛出**，不可当作“模型学会避障”的证据（会偏置模态与推理成本）。
  - `MacroSkel (offroad_weight=0.1)` 在一次尝试中出现 **灾难性退化**（`collision_rate_any=1.0`，`waypoint_offroad_rate≈0.955`）：
    - 这通常意味着：训练失稳 / 配置错误 / 数据版本不一致，而不是“约束项无效”。
    - 新版 `train_macro_diffusion.py` 已在训练日志中加入 `gt_offroad_pen_w`（GT 线性骨架在该 proxy 下的惩罚下界）用于定位：若 `gt_offroad_pen_w` 很大，先检查 `count_thr` 与 drivable proxy 是否一致；若 GT 很小但 pred 很大，才是模型没学会“看图”。

- Phase 3（Macro Hard Support + AR，自回归连通性写回模型，detour-hard，N=400，K=1，**无事后筛选**）：
  - 旧版（并行预测 wp1/wp2/end）：
    - `WP_ANY=0` 但 `CUT/COLL≈0.145~0.176`（主要碰撞来自折线切墙，尤其 wp2→end 段）。
  - AR 版（顺序生成 `wp1→wp2→end`，并把前序 one-hot map 作为条件）：
    - `argmax (K=1)`：`collision_rate_any=0.060`，`cut_only_rate=0.060`，`WP_ANY=0`（seg0/1/2=0.0425/0.0225/0.0125）
    - `multinomial (K=1)`：`collision_rate_any=0.085`，`cut_only_rate=0.085`，`WP_ANY=0`（seg0/1/2=0.0600/0.0250/0.0175）
    - 结论：**G1 已通过**（1-shot，非 best-of-k），瓶颈从“点越界”转为“mask 内分布是否合理 + G2 真实性”。

---

## Phase 3（Macro Hard Support）离线审计（先做这 4 项，再写模型代码）

> 目的：把 PI review 里提到的“量化误差/空 mask/coarse recall/输入相关性”用 CPU 离线审计一次性量化，避免边写边改走弯路。

前置：准备一个包含 `start_pos/targets` 的 windows 文件（推荐直接复用你跑 macro 采样时的 `samples.npz`，例如 `data/experiments/phys_*detourhard*/samples.npz`）。

```bash
export PROC=data/processed_passenger_dt30
export NAV=$PROC/nav_field.npz
export IN=data/experiments/phys_mapuse_test_none/samples.npz   # <-- 改成你实际的 windows npz

python -m src.evaluation.macro_hardsupport_offline_audit \
  --in_samples_npz "$IN" \
  --nav_file "$NAV" --count_thr 1.0 \
  --patch_size 64 --coarse_g 16 \
  --sample_step 0.5 --max_samples_per_segment 256 \
  --out_dir data/experiments/macro_hardsupport_offline_audit \
  --out_json data/experiments/macro_hardsupport_offline_audit/report.json
```

**只看 3 个输出就够**：
- `nav_stats.empty_strict_patch_rate`：若非 0，需要定义 empty-mask fallback（skip / 回退到更宽松 mask / 全局投影）。
- `gate.oracle_proj.cut_only_rate`：这是 `WP_ANY=0` 时的 CUT 下界（会随子集/N/采样细节波动；以你本次跑出的 gate JSON 为准，避免把某次快照数字写死）。
- `gate.oracle_proj_coarse_only.cut_only_rate`：如果 coarse-only 也能 <0.10，可以考虑先不做 Stage2（更 KISS）；否则必须做 pixel-level（或 fine stage）。

---

## 争议点记录（PI Review）：Hard Support 会不会让 G2 失去意义？

> 目的：把争议说清楚、把结论落到“可执行的审计”，避免团队在同一问题上反复循环。

### 事实（硬成立）

- Hard Support（masked softmax）把 Macro 的输出 support 限制在 `mask` 内，因此：
  - `WP_ANY=0` 主要来自**结构约束**（不是模型“学会道路识别”）。
- Hard Support **不是** test-time 事后筛选（不是 rejection/best-of-k）：
  - 模型从一开始就建模 `p(z | cond, mask)`，而不是先采样 `p(z|cond)` 再丢掉不合法样本。
  - 因此不存在“先生成一堆非法 → 事后擦屁股造成 selection bias”的伪提升。

### PI 的核心担忧（必须正面回应）

1) **特征层面未必学到道路语义（L1）**  
模型 backbone 会看全 patch，但输出层只允许在 mask 内输出；若 mask 漏路，模型也无法“发现”漏掉的路。

2) **Hard Support 与软语义（软约束）不是一回事**  
Hard Support 解决“能不能”（合法性/支持集），软语义解决“选哪个更合理”（在 mask 内的概率质量分配、连通性、绕路）。

3) **Raw vs Proj GT 的审计不够**  
仅比较 Raw/Proj GT 只能发现“mask 剪掉 GT 太多”，不能发现“mask 内的分布是否错位/是否中心偏置”。

4) **论文定位的 trade-off 必须写清楚**  
接受 Hard Support 作为建模假设 → 论文应定位为 **map/weak-map 条件下的规划/预测**；  
若要 claim “道路识别（L1）”，必须引入 OSM/分割监督/语义地图（另起工作量级）。

### 当前裁决（主线推进口径）

- 主线暂不做 L1（道路识别）。我们把 `mask` 视为 weak-map（外部输入），目标是验证 **trip-level 决策 + micro 执行** 是否能生成真实轨迹。
- 为了避免“mask 内乱选/偏置”导致 G2 结论被质疑：**必须增加“mask 内分布对齐审计”**（见下一节）。

---

## 新增审计：Mask 内分布对齐（用于支撑 G2 的可信性）

> 目标：回答“模型是否真的在 mask 内学到了合理分布”，而不是靠硬约束掩盖问题。

### 审计输出（KISS：只看这三类）

- **Heatmap 对齐（每个点：wp1/wp2/end）**：Pred vs GT 的 patch-heatmap 距离（JSD），只在 `mask==1` 的像素上归一化。
- **Clearance 对齐**：点到最近 offroad 的距离分布（Pred vs GT）。
- **中心偏置**：点到 patch center（start）距离分布（Pred vs GT），用于检测“都往安全中心挤”。

### 运行命令（CPU-only）

前置：你需要一个包含 `start_pos/targets/z_k_grid` 的 macro 采样文件（例如 `dump_macro_hardsupport_ar_samples.py` 的输出）。

```bash
export NAV=data/processed_passenger_dt30/nav_field.npz
export IN=data/experiments/<your_macro_samples>/samples.npz

python -m src.evaluation.macro_mask_alignment \
  --samples_npz "$IN" \
  --nav_file "$NAV" --count_thr 1.0 \
  --patch_size 64 \
  --quiet \
  --out_json data/experiments/<your_macro_samples>/mask_alignment.json \
  --out_png  data/experiments/<your_macro_samples>/mask_alignment.png
```

**关键判读**（建议写进汇报）：
- 若 Pred vs GT 的 heatmap-JSD/clearance-JSD/center-JSD 都显著低于“随机 baseline”（脚本会给出），说明模型在 mask 内并非乱选。
- 若 Pred 明显更贴边（clearance 更小）或明显更保守（clearance 更大/更中心），需在 G2 讨论中解释“可行性 vs 多样性”的偏置来源。

### 已跑事实快照（detour-hard，AR sample1，N=400，K=1）

- `valid_rates`：`pred_valid_rate=1.000`，`gt_raw_valid_rate=0.962`，`gt_proj_valid_rate=0.985`，`avg_drivable_pixels_per_patch≈2662.9/4096`
- `heatmap_jsd_pref (pred vs gt_proj)`：`wp1=0.4796`，`wp2=0.6360`，`end=0.7903`
  - 对照 random baseline：`wp1≈0.9052`，`wp2≈0.9076`，`end≈0.9143`（显著更差）
  - 结论：模型在 mask 内**不是随机选点**，但 end 的像素级偏好仍与 GT 有较大差距（G2 需要重点关注 end 相关的 detour/topology）。

---

## Phase 3（Macro Hard Support AR）G2：Macro→Skeleton→DetRes（K=1，无筛选）

> 目的：在 **不做任何 best-of-k / rejection** 的前提下，验证端到端轨迹（Macro 决策 + Micro 执行）是否真实/合理。
> 
> 口径：detour-hard 子集，`K=1`（分别跑 `argmax` 与 `multinomial`）。

### Step 1：采样 Macro（同时输出轨迹：skeleton-only）

> 使用 `--emit_traj`：脚本会把 `z_k_grid` 直接接到 `skeleton prior`，输出 `preds/preds_k`，可直接跑 `detour_validity/plot_physical_stats`。

```bash
export PROC=data/processed_passenger_dt30
export DATA=$PROC/trajectories/shenzhen_trajectories.h5
export NAV=$PROC/nav_field.npz

# 你的 MacroHardSupportAR 训练目录名（data/experiments/<EXP>/last.pt）
export EXP=macro_hardsupport_ar_p64_thr1_s0

# detour-hard windows
export WINS=data/experiments/gt_passenger_dt30_test/test_detour_hard_top10.npz

# ---- argmax (K=1) ----
export OUT_SKEL=phys_macro_hardsupport_ar_detourhard_${EXP}_skel_argmax
python -m src.evaluation.dump_macro_hardsupport_ar_samples \
  --exp_name "$OUT_SKEL" \
  --checkpoint "data/experiments/$EXP/last.pt" \
  --data_path "$DATA" --nav_file "$NAV" --split test \
  --obs_len 8 --pred_len 12 \
  --patch_size 64 --count_thr 1.0 \
  --k_samples 1 --sample_mode argmax \
  --emit_traj \
  --save_samples 400 --max_batches 13 --batch_size 32 --num_workers 8 --seed 0 \
  --windows_npz "$WINS"
```

### Step 2：同一套 Macro 样本接 DetRes executor（macro+micro 闭环）

```bash
export MICRO=data/experiments/phys_oracleWP_detres_k2/last.pt
export OUT_DETRES=phys_macro_hardsupport_ar_detourhard_${EXP}_detres_argmax
python -m src.evaluation.dump_macro_hardsupport_ar_samples \
  --exp_name "$OUT_DETRES" \
  --checkpoint "data/experiments/$EXP/last.pt" \
  --micro_checkpoint "$MICRO" \
  --data_path "$DATA" --nav_file "$NAV" --split test \
  --obs_len 8 --pred_len 12 \
  --patch_size 64 --count_thr 1.0 \
  --k_samples 1 --sample_mode argmax \
  --emit_traj \
  --save_samples 400 --max_batches 13 --batch_size 32 --num_workers 8 --seed 0 \
  --windows_npz "$WINS"
```

> 若要跑 `multinomial`（K=1）：把两条命令里的 `--sample_mode argmax` 改成 `--sample_mode multinomial`，输出目录建议用后缀 `_sample1`。

### Step 3：G2 评估（拓扑 + 物理纹理）

```bash
# detour_validity：建议 detour_pct=100，把整个 detour-hard 子集当 detour 来检验稳定性
python -m src.evaluation.detour_validity \
  --inputs "MacroSkel:data/experiments/$OUT_SKEL/samples.npz" \
           "Macro+DetRes:data/experiments/$OUT_DETRES/samples.npz" \
  --ds 0.5 --lags 1 2 4 8 --offset_fracs 0 0.25 0.5 0.75 \
  --detour_pct 100 --bootstrap 200 --noise_splits 200 \
  --out_json data/experiments/phys_macro_hardsupport_ar_detourhard_g2_detour_validity.json

# physical_stats：看 speed/accel/turn 分布与 DCV
python -m src.visualization.plot_physical_stats \
  --inputs "MacroSkel:data/experiments/$OUT_SKEL/samples.npz" \
           "Macro+DetRes:data/experiments/$OUT_DETRES/samples.npz" \
  --turn_min_speed 0.1 --dcv_speed_pctl 99.5 --dcv_accel_pctl 99.5 \
  --save_metrics --output_dir essay/figures/physical_stats \
  --stem fig_physical_stats_macro_hs_ar_g2
```

### Step 3.1（必做、补齐 PI 口径）：dev/len 的“方向性”审计（Pred 更直还是更绕？）

> 背景：`detour_validity` 里 `JSD_max_dev_ratio/JSD_len_ratio` 只回答“和 GT 分布差多少”，**不回答偏差方向**（更直/更绕）。
> 
> 这个审计会同时打印 GT 与 Pred 的 raw 分位数，并输出 `Δp50`（Pred - GT）：
> - `Δ>0`：Pred 比 GT 更绕/更长（更 detour）
> - `Δ<0`：Pred 比 GT 更直/更短（更 straight）

```bash
python -m src.evaluation.detour_scalar_direction_audit \
  --inputs "MacroSkel:data/experiments/$OUT_SKEL/samples.npz" \
           "Macro+DetRes:data/experiments/$OUT_DETRES/samples.npz" \
  --detour_pct 100 \
  --quiet \
  --out_json data/experiments/phys_macro_hardsupport_ar_detourhard_g2_scalar_direction.json
```

### Step 3.2（必做、补齐 PI 口径）：为什么 ORACLE_PROJ 也有 ~7% CUT？

> 目的：区分两种根因
> - **Mask 过严/有孔洞**：轻微膨胀 drivable 区域就能显著降低 CUT
> - **Straight-line skeleton 上限**：即便膨胀，CUT 仍不怎么降（道路本身是弯的，直线连点天然切角）
> 
> 这里做两步：先导出 Oracle raw/proj（K=1）→ 再做 dilation sensitivity 审计。

```bash
export NAV=data/processed_passenger_dt30/nav_field.npz
export GTW=data/experiments/gt_passenger_dt30_test/test_detour_hard_top10.npz

# Oracle RAW（不投影）
python -m src.evaluation.oracle_macro_z_gate \
  --in_samples_npz "$GTW" \
  --nav_file "$NAV" --count_thr 1.0 \
  --out_npz data/experiments/gt_passenger_dt30_test/oracle_macro_z_raw.npz

# Oracle PROJ（投影到 strict drivable）
python -m src.evaluation.oracle_macro_z_gate \
  --in_samples_npz "$GTW" \
  --nav_file "$NAV" --count_thr 1.0 \
  --project_strict \
  --out_npz data/experiments/gt_passenger_dt30_test/oracle_macro_z_proj.npz

# Dilation sensitivity（关键审计）
python -m src.evaluation.oracle_cut_cause_audit \
  --samples_npz data/experiments/gt_passenger_dt30_test/oracle_macro_z_proj.npz \
  --nav_file "$NAV" --count_thr 1.0 \
  --dilate_iters 0 1 2 \
  --quiet \
  --out_json data/experiments/gt_passenger_dt30_test/oracle_cut_cause_audit.json
```

### Step 4（可选、快速）：END 是否真的利用了 trip destination？

> 用 `end` 的“向目的地靠近进展”做一个最小审计：如果与随机 baseline 接近，说明 end 没用好 destination（建议先做 destination conditioning 的可辨识增强；语义/遥感属于另一条假设，需要单独做 ablation）。

```bash
python - <<'PY'
import os
import numpy as np
from pathlib import Path
from src.features.nav_field import NavField

nav = NavField(os.environ.get("NAV", "data/processed_passenger_dt30/nav_field.npz"))
in_npz = Path(os.environ.get("IN", "data/experiments/phys_macro_hardsupport_ar_detourhard_macro_hardsupport_ar_p64_thr1_s0_argmax/samples.npz"))
count_thr = float(os.environ.get("COUNT_THR", "1.0"))
patch_size = int(os.environ.get("PATCH", "64"))
r = patch_size // 2

d = np.load(str(in_npz), allow_pickle=True)
start = np.asarray(d["start_pos"], np.float32)
dest = np.asarray(d["dest_pos"], np.float32)
z = np.asarray(d["z_k_grid"], np.float32)  # (N,K,3,2)
end = z[:, 0, 2]  # (N,2)

dist0 = np.linalg.norm(start - dest, axis=-1)
dist1 = np.linalg.norm(end - dest, axis=-1)
prog = dist0 - dist1

# random baseline: uniform over drivable pixels in each patch
rand_end = np.zeros_like(end)
rng = np.random.default_rng(0)
for i in range(start.shape[0]):
    patch = nav.get_patch(start[i], patch_size=patch_size, channel2="count")
    drv = patch[2] >= count_thr
    ys, xs = np.where(drv)
    if ys.size == 0:
        ys, xs = np.where(np.ones((patch_size, patch_size), dtype=bool))
    j = int(rng.integers(0, ys.size))
    center = np.floor(start[i]).astype(np.float32)
    rand_end[i] = center + np.array([ys[j] - r, xs[j] - r], dtype=np.float32)

dist1r = np.linalg.norm(rand_end - dest, axis=-1)
progr = dist0 - dist1r

def q(x):
    return np.percentile(x, [10, 50, 90]).tolist()

print("N:", int(start.shape[0]))
print("PROGRESS (model)  p10/p50/p90:", q(prog))
print("PROGRESS (random) p10/p50/p90:", q(progr))
print("MEAN progress model/random:", float(np.mean(prog)), float(np.mean(progr)))
PY
```

### Step 4.1（必做、补齐 PI 口径）：END “不精”到底是哪种不精？（距离/走廊/像素）

> 目的：把“方向对但落点不精”拆成可行动的三类（对应不同修复方向）：
> - `dist_error`：沿 start→dest 方向的距离偏差大（可能需要更强 destination conditioning）
> - `corridor_error`：横向偏差大（可能是选错平行道路/走廊，可能需要更强拓扑/语义）
> - `both_error`：两者都有
> 
> 注意：这里的 “GT end” 默认取 window end（`targets[-1]`），可选 `--use_gt_proj` 投影到 strict drivable 后再对齐口径。

```bash
python -m src.evaluation.end_imprecision_audit \
  --samples_npz "data/experiments/$OUT_SKEL/samples.npz" \
  --nav_file "$NAV" --count_thr 1.0 \
  --use_gt_proj \
  --thr_along 8 --thr_cross 4 \
  --quiet \
  --out_json data/experiments/phys_macro_hardsupport_ar_detourhard_end_imprecision.json
```

## 主线三步 Go/No-Go（最短实验序列）

> 这三步的设计目标是：**先把积分打开（Oracle z）**，把风险压到最低，再决定是否进入训练。

### 准备：导出 GT windows 为 `samples.npz`（CPU-only）

本协议默认所有 gate/validity 都以**同一份 GT windows**为基准，避免“拿模型输出当 GT”导致的同义反复。

```bash
python -m src.evaluation.dump_gt_windows_npz \
  --processed_dir data/processed_passenger_dt30 \
  --split test \
  --out_npz data/experiments/gt_passenger_dt30_test/samples.npz \
  --obs_len 8 --pred_len 12 \
  --num_samples 20000 \
  --seed 0
```

输出字段（最小闭环）：`origin_pos/dest_pos/start_pos/targets/traj_idx/start_t`。

> Sanity：若后续 `detour_validity` 报 `GT mismatch`，基本意味着不同方法的 `samples.npz` 不是同一批窗口。请先确认输入文件都含 `traj_idx/start_t`（用于对齐）；新版脚本会自动写入，并在报告里给出 `stats.alignment`（若发生对齐/丢弃非交集窗口）。

### Go/No-Go 1：OD 条件下是否存在“稳定多模态混合”？

**目的**：确认直线坍缩是否真的来自“混合分布的梯度平均”。  
**做法**（GT-only，CPU-only）：把相近 OD 做桶（需要 coarse bin 才有足够重复），对桶内 GT 轨迹做粗尺度几何聚类/分裂，检查是否出现稳定的 2+ 簇。

- 脚本：`src/evaluation/od_multimodality_gate.py`
- 推荐做法：扫一遍 `od_bin`（例如 8/12/16），看结论是否稳定；`od_bin` 越小越“严格”（但重复越少）。

```bash
python -m src.evaluation.od_multimodality_gate \
  --samples_npz data/experiments/gt_passenger_dt30_test/samples.npz \
  --od_bin 12 --min_bucket_n 50 --sep_thr 2.0 \
  --out_json data/experiments/gt_passenger_dt30_test/od_gate_dest_odbin12.json
```

**判读**（经验口径，避免过拟合阈值）：
- 若在相对严格的 `od_bin`（如 8/12）下，多模态桶占比接近 0：说明“同 OD 多路线混合”不是主要矛盾，分层仍可做但预期收益变小（更可能是 micro/可行域问题）。
- 若在较严格 bin 下仍存在稳定多模态桶：强证据支持“混合导致平均场”解释，分层值得作为主线。

### Go/No-Go 2：Oracle waypoints 的 skeleton-only 能否承载 coarse 拓扑模态？

**目的**：验证 “少量 waypoint 能表达拓扑” 这个压缩瓶颈是否成立。  
**做法**：从每条 GT 提取 2–3 个 waypoint（建议 RDP fixed-K），生成 skeleton，并用 Phase 0 的空间尺度指标评估（不看 MSE）。

1) 先做 **硬可行性**：`Waypoint Gate`（碰撞率/越界率）  
2) 再做 **coarse detour 指标对齐**：`detour_validity`（overall + detour subset）

生成 skeleton-only 的 `samples.npz`（便于统一后续评估）：
```bash
python -m src.evaluation.make_oracle_skeleton \
  --samples_npz data/experiments/gt_passenger_dt30_test/samples.npz \
  --out_npz data/experiments/gt_passenger_dt30_test/skeleton_rdp_k2_linear.npz \
  --waypoint_mode rdp_dev --num_waypoints 2 --skeleton linear
```

对照（0 waypoint 的 straight skeleton）：
```bash
python -m src.evaluation.make_oracle_skeleton \
  --samples_npz data/experiments/gt_passenger_dt30_test/samples.npz \
  --out_npz data/experiments/gt_passenger_dt30_test/skeleton_straight_k0_linear.npz \
  --waypoint_mode time --num_waypoints 0 --skeleton linear
```

硬可行性（Waypoint Gate：碰撞率/越界率，<10% 才讨论后续）：
```bash
python -m src.evaluation.waypoint_gate \
  --samples_npz data/experiments/gt_passenger_dt30_test/samples.npz \
  --nav_file data/processed_passenger_dt30/nav_field.npz \
  --waypoint_mode rdp_dev --num_waypoints 2 \
  --skeleton linear \
  --out_json data/experiments/gt_passenger_dt30_test/waypoint_gate_rdp_k2.json
```

评估（空间尺度 turn + max_dev_ratio + len_ratio；含 CI+noise floor）：
```bash
python -m src.evaluation.detour_validity \
  --inputs "StraightK0:data/experiments/gt_passenger_dt30_test/skeleton_straight_k0_linear.npz" \
           "RDPK2:data/experiments/gt_passenger_dt30_test/skeleton_rdp_k2_linear.npz" \
  --ds 0.5 --lags 1 2 4 8 --offset_fracs 0 0.25 0.5 0.75 \
  --detour_pct 10 --bootstrap 200 --noise_splits 200 \
  --out_json data/experiments/gt_passenger_dt30_test/detour_validity_skeleton_k2.json
```

**Go/No-Go**：
- 若 `RDPK2` 在 detour subset 上显著优于 `StraightK0`，且 `Waypoint Gate` 的碰撞率 < 10%：通过（waypoint 瓶颈成立）。
- 若 skeleton-only 仍对不齐 coarse detour（或碰撞率高）：说明 “少点 waypoint 承载拓扑” 不成立，应改 K/改表征或直接转 weak-map/graph。

### Go/No-Go 3：Oracle waypoints 条件下，micro 是否“可执行且不破坏拓扑”？

**目的**：回答“分层能不能修复不物理”的关键。  
**必须三对照**（同一批 Oracle waypoints / 同一 detour subset）：
1) skeleton-only（不训练）
2) deterministic residual（小网络，先证伪 residual 任务不可学）
3) diffusion residual（主方案）

**硬条件（必须同时满足）**：
- (A) coarse 拓扑不被破坏：final 的 `turn@L / max_dev_ratio / len_ratio` **不差于 skeleton-only**
- (B) 物理纹理显著改善：`speed/accel/DCV`（或更强的 kinematic 耦合指标）显著优于 skeleton-only

若 (A) 失败：更可能是 micro 表示/条件注入问题（或 residual 偷跑 macro）；不要继续堆层级。  
若 (A) 成功但 (B) 失败：说明分层拆拓扑是对的，但 micro 物理建模不足（需要 Frenet residual / band-limit / kinematic 约束等）。

**可直接运行的最小命令（Go/No-Go 3）**（注意：训练/评估需要 torch 环境；绘图/统计可在 `emotion` 跑）：

0) 统一数据路径（按你的实际目录改 `PROC` 即可）：
```bash
export PROC=data/processed_passenger_dt30
export DATA=$PROC/trajectories/shenzhen_trajectories.h5
export NAV=$PROC/nav_field.npz
```

1) `skeleton-only`（oracle waypoints + linear + arclen resample；用于对照，**不训练**）：
```bash
python -m src.evaluation.dump_skeleton_prior_samples \
  --exp_name phys_oracleWP_skeleton_k2 \
  --model_type physics \
  --data_path $DATA --nav_file $NAV \
  --split test --obs_len 8 --pred_len 12 \
  --batch_size 32 --num_workers 8 \
  --save_samples 400 --max_batches 13 --seed 0
```

2) `deterministic residual`（SeqBaseline 预测 residual=GT−skeleton；快速证伪 residual 任务不可学）：
```bash
python -m src.training.train_baseline \
  --exp_name phys_oracleWP_detres_k2 \
  --data_path $DATA --split train \
  --obs_len 8 --pred_len 12 \
  --batch_size 256 --num_workers 8 \
  --hidden_dim 128 --epochs 20 --max_batches 200 \
  --cond_mode oracle_wp_end --waypoint_mode rdp_dev --num_waypoints 2 \
  --prior_mode skeleton_wp --seed 0

python -m src.training.evaluate \
  --exp_name phys_oracleWP_detres_k2_eval \
  --model_type baseline \
  --data_path $DATA --split test \
  --checkpoint data/experiments/phys_oracleWP_detres_k2/last.pt \
  --cond_mode oracle_wp_end --waypoint_mode rdp_dev --num_waypoints 2 \
  --prior_mode skeleton_wp \
  --batch_size 32 --num_workers 8 \
  --save_samples 400 --max_batches 13 --samples_only --seed 0
```

3) `diffusion residual`（主方案：PhysicsConditionDiffusion 预测 residual=GT−skeleton）：
```bash
python -m src.training.train_diffusion \
  --exp_name phys_oracleWP_diffres_k2 \
  --model_type physics \
  --data_path $DATA --nav_file $NAV --split train \
  --obs_len 8 --pred_len 12 \
  --batch_size 128 --num_workers 8 \
  --hidden_dim 128 --epochs 20 --max_batches 200 \
  --diff_steps 100 --pred_type eps \
  --cond_mode oracle_wp_end --waypoint_mode rdp_dev --num_waypoints 2 \
  --prior_mode skeleton_wp --seed 0

python -m src.training.evaluate \
  --exp_name phys_oracleWP_diffres_k2_eval \
  --model_type physics \
  --data_path $DATA --nav_file $NAV --split test \
  --checkpoint data/experiments/phys_oracleWP_diffres_k2/last.pt \
  --cond_mode oracle_wp_end --waypoint_mode rdp_dev --num_waypoints 2 \
  --prior_mode skeleton_wp \
  --batch_size 32 --num_workers 8 \
  --num_samples_per_condition 20 --save_all_k \
  --diff_steps 100 \
  --save_samples 400 --max_batches 13 --samples_only --seed 0
```

4) 指标/作图（同一批 windows，三对照；先看 (A) 再看 (B)）：
```bash
python -m src.evaluation.detour_validity \
  --inputs "Skeleton:data/experiments/phys_oracleWP_skeleton_k2/samples.npz" \
           "DetRes:data/experiments/phys_oracleWP_detres_k2_eval/samples.npz" \
           "DiffRes:data/experiments/phys_oracleWP_diffres_k2_eval/samples.npz" \
  --ds 0.5 --lags 1 2 4 8 --offset_fracs 0 0.25 0.5 0.75 \
  --detour_pct 10 --bootstrap 200 --noise_splits 200 \
  --out_json data/experiments/phys_oracleWP_go_nogo3_detour_validity.json

python -m src.visualization.plot_physical_stats \
  --inputs "Skeleton:data/experiments/phys_oracleWP_skeleton_k2/samples.npz" \
           "DetRes:data/experiments/phys_oracleWP_detres_k2_eval/samples.npz" \
           "DiffRes:data/experiments/phys_oracleWP_diffres_k2_eval/samples.npz" \
  --use_all_k --k_max 10 \
  --turn_min_speed 0.1 --dcv_speed_pctl 99.5 --dcv_accel_pctl 99.5 \
  --save_metrics --output_dir essay/figures/physical_stats \
  --stem fig_physical_stats_go_nogo3_oracleWP
```

---

## 0) 三条必须写死的硬条款（写进所有计划/README）

1. **多尺度以空间尺度为主**：弧长重采样/插值 + spatial lag（而非 time stride），并做 **multi-offset 聚合**（消除 alias/相位偏置）。
2. **detour 判据至少二元联合**：粗尺度转向（turn@L） + `max deviation`（或 `length ratio`）同时报告，防止“靠乱转角度”骗过 turn 直方图。
3. **Phase 2 必须有三对照**：`skeleton-only` vs `deterministic residual` vs `diffusion residual`，且 residual 需结构性约束防止偷跑 macro（否则失败无法归因）。

---

## 1) 跨 Phase 最容易走偏的 6 个漏洞（以及补丁）

### 漏洞 1：time stride ≠ 空间尺度
- 速度分布不一致时，stride 低通掉的是“时间高频”，不等价于“几何高频”，会误判 detour。
- **补丁**：统一用弧长参数化，再在空间 lag（如 L=1/2/4/8）上算 turn/曲率分布。

### 漏洞 2：stride 的相位偏置（alias + offset bias）
- stride=4 只取 offset=0 会系统性跳过转折点。
- **补丁**：对每个尺度做 **multi-offset 聚合**（offset=0..stride-1 或 ds 的分数偏移）。

### 漏洞 3：TurnAngle 直方图不看序列结构，可被“乱转圈”骗过
- **补丁**：至少加一个几何位移类量：`max lateral deviation`（建议归一化到 chord）或 `path length ratio`。

### 漏洞 4：JSD 口径不锁死（bin/样本量/圆周变量）
- turn-angle 的直方图 bin 边界会显著影响数值；样本少时方差更大。
- **补丁（Phase 0 写死）**：
  - 固定 turn 的 bin（例如 `[0,π]` 等宽），并把 bin edges 写入 JSON；
  - 用 bootstrap 给 95% CI；
  - 计算 `GT vs GT (split-half)` 的 JSD 作为 **噪声地板**。

### 漏洞 5：detour 是低频且低频事件（尾部稀缺），overall 会被淹没
- **补丁**：定义 `detour subset`（只用于评估），所有 Go/No-Go 必须同时报告：
  - overall
  - detour subset

### 漏洞 6：用 KDE/密度当 “feasibility gate” 会把 detour 过滤掉
- KDE 是典型性过滤，尾部 detour 很可能被压死。
- **补丁**：
  - gate 只做硬可行性（速度/加速度界、越界/飞天等），不要把训练密度当可行性；
  - 若必须用密度：降级为 soft penalty，并在 detour subset 上单独报告通过率。

---

## 2) Phase 0：Detour Validity 指标体系（CPU-only）

### 2.1 指标（固定口径）

1) **多尺度转向（空间尺度）**  
给定空间 lag `L`：
\[
\Delta\theta^{(L)} = \left|\mathrm{wrap}\big(\angle(p(s+L)-p(s))-\angle(p(s)-p(s-L))\big)\right|
\]
统计 `Δθ(L)` 的分布，用固定 bin 的 `JSD` 衡量 Pred vs GT。

2) **Max deviation（防作弊）**  
相对 chord（start→end）最大横向偏离，并用 chord 长度归一化：
`max_dev_ratio = max_dev / chord_len`

3) **Length ratio（防回环/乱转）**  
`len_ratio = path_len / chord_len`

> 注意：turn 分布“像”不代表拓扑正确；`max_dev_ratio/len_ratio` 是最便宜的反作弊约束。

### 2.2 统计流程（可止损）
- 固定 bin（turn 使用 `[0,π]` 等宽；标量 bin 由 GT 分位数确定并写入报告）。
- bootstrap 得到 95% CI。
- **噪声地板**：`GT split-half` 的 JSD（同口径）作为 floor。

### 2.3 detour subset（只评估）
- 默认用 GT 的 `max_dev_ratio` 取 top-10%（或按需求调阈值），并写入阈值与子集大小。

### 2.4 工具脚本

- 脚本：`src/evaluation/detour_validity.py`

示例：
```bash
python -m src.evaluation.detour_validity \
  --inputs "Prior:data/experiments/prior_geo_density_test/samples.npz" \
           "Ours(CFG2):data/experiments/phys_cfg2_geo_density_test/samples.npz" \
  --ds 0.5 --lags 1 2 4 8 --offset_fracs 0 0.25 0.5 0.75 \
  --detour_pct 10 --detour_score max_dev_ratio \
  --bootstrap 200 --noise_splits 200 \
  --out_json data/experiments/detour_validity/prior_vs_cfg2.json
```

---

## 3) Phase 1：Waypoints / Skeleton 是否能承载拓扑模态（不看 MSE）

核心判据：在 detour subset 上，`skeleton-only` 是否能对齐：
- `turn@L(粗尺度)` 分布
- `max_dev_ratio` 分布
- `len_ratio` 分布

补强建议（仍然不烧卡）：
- 时间分位点基线改为 **弧长分位点**（避免速度污染）。
- `max_dev` 选点建议用固定-K 的 RDP 两步版（S 型 detour 至少需要 2 个点）。
- 必须包含对照：**0 waypoint（仅 start/end）** 的 skeleton。

同时跑 `Waypoint Gate` 的硬可行性（碰撞率/越界率），避免 coarse 表征本身不可行：  
见 `src/evaluation/waypoint_gate.py`。

---

## 4) Phase 2：Micro-only Executability（最关键止损点）

必须三对照（同一批 Oracle waypoints / 同一 detour subset）：
1) `skeleton-only`（不训练）
2) `deterministic residual`（小网络，先证伪 “residual 任务不可学”）
3) `diffusion residual`（主方案）

结构性补丁（避免 residual 偷跑 macro 或被锁死）：
- residual 建议在 **Frenet frame** 里生成（法向偏移为主），并在 waypoint 处施加边界条件（残差为 0）。
- 或显式去掉 residual 的低频分量（band-pass），保证拓扑只能由 skeleton 承担。

止损线（示例口径）：
- detour subset 上：最终轨迹的 `turn@L(粗尺度)` **不能差于 skeleton-only**（不能被 residual 搞坏）
- 同时：`speed/accel`（或微观纹理指标）应显著优于 skeleton-only（证明 micro 不是摆设）

---

## 5) Phase 3：Macro waypoint 可学性（仅在 Phase 2 通过后）

评估不要只看 best-of-K（容易被 K 大小骗）：
- coverage（best-of-K 命中）
- **detour subset 的概率质量/占比**（detour 模态是否真的被分配到足够概率）

训练侧避免尾部被忽略：
- detour subset oversample / reweight（成本敏感）
- feasibility gate 不要用 KDE 硬过滤尾部

### Phase 3 最小可证伪实现（Macro Diffusion + 弱图，不烧卡）

> 目标：用 **Macro Diffusion（低维 z）** 回答 “z（waypoints+end_anchor）是否可从 (obs, trip_od, nav_patch[count]) 学到多模态分布”。  
> 关键：把不确定性放在 Macro；micro 由 DetRes 确定性执行。  
> 采样预算：`K=20, diff_steps=20`（6D/3-step 低维空间，训练与采样都很快）。

0) 统一数据路径：
```bash
export PROC=data/processed_passenger_dt30
export DATA=$PROC/trajectories/shenzhen_trajectories.h5
```

1) 生成 detour-hard 测试子集（避免 detour 被 overall 淹没；后续 G2 建议直接在该子集上跑 `detour_validity --detour_pct 100`）：
```bash
python -m src.evaluation.make_detour_hard_subset \
  --in_npz data/experiments/gt_passenger_dt30_test/samples.npz \
  --out_npz data/experiments/gt_passenger_dt30_test/test_detour_hard_top10.npz \
  --score max_dev_ratio --top_pct 10 --max_n 5000 --seed 0
```

2) 训练 Macro Diffusion（监督信号来自 `oracle_wp_end`，输入：`obs + trip_od + nav_patch(count,current_only)`）：
```bash
export NAV=$PROC/nav_field.npz

python -m src.training.train_macro_diffusion \
  --exp_name macro_zdiff_k2_count_current \
  --data_path $DATA --nav_file $NAV --split train \
  --obs_len 8 --pred_len 12 \
  --patch_size 32 --nav_patch_channel2 count \
  --hidden_dim 128 --diff_steps 20 --pred_type eps \
  --batch_size 512 --num_workers 8 \
  --epochs 20 --max_batches 200 --seed 0 \
  --count_thr 1.0 --log_every 100

# 若 MacroSkel 的 G1 碰撞率过高（比如 >10%）：建议加训练期 OffRoad Penalty（不改架构）
# 该项用 nav_count 在 start->wp1->wp2->end 折线上做可微采样，惩罚 count<thr（近似 G1 gate）。
# 注意：OffRoad Penalty 默认按 alpha^2(=alphas_cumprod[t]) 做加权，避免在高噪声 timestep 上惩罚噪声主导的 x0_pred（防止训练不稳定）。
python -m src.training.train_macro_diffusion \
  --exp_name macro_zdiff_k2_count_current_offroad0p1 \
  --data_path $DATA --nav_file $NAV --split train \
  --obs_len 8 --pred_len 12 \
  --patch_size 32 --nav_patch_channel2 count \
  --hidden_dim 128 --diff_steps 20 --pred_type eps \
  --batch_size 512 --num_workers 8 \
  --epochs 20 --max_batches 200 --seed 0 \
  --count_thr 1.0 --offroad_weight 0.1 --offroad_samples_per_segment 16 \
  --log_every 100

# 若出现“penalty 下降但 gate 仍高”的 proxy gap：把聚合从 mean 切到更贴近 ANY/OR 的风险敏感形式。
# - mean：平均违例（可能无法压低 collision_rate_any）
# - max/lse：最坏点违例（更贴近 gate 的 any-point 碰撞判据）
python -m src.training.train_macro_diffusion \
  --exp_name macro_zdiff_k2_count_current_offroad0p1_lse \
  --data_path $DATA --nav_file $NAV --split train \
  --obs_len 8 --pred_len 12 \
  --patch_size 32 --nav_patch_channel2 count \
  --hidden_dim 128 --diff_steps 20 --pred_type eps \
  --batch_size 512 --num_workers 8 \
  --epochs 20 --max_batches 200 --seed 0 \
  --count_thr 1.0 --offroad_weight 0.1 --offroad_samples_per_segment 16 \
  --offroad_agg lse --offroad_lse_beta 10 \
  --log_every 100

# 若确认瓶颈在 “nav_patch -> 向量” 的信息塌缩：启用 Scheme-3（ControlNet 式多尺度注入），
# 让 UNet1D 在每个 down-block 都能获得来自 nav_patch 的控制信号（zero-init，训练时逐步学会“看墙/看路”）。
python -m src.training.train_macro_diffusion \
  --exp_name macro_zdiff_k2_count_current_controlnet_offroad0p1_lse \
  --data_path $DATA --nav_file $NAV --split train \
  --obs_len 8 --pred_len 12 \
  --patch_size 32 --nav_patch_channel2 count \
  --hidden_dim 128 --diff_steps 20 --pred_type eps \
  --batch_size 512 --num_workers 8 \
  --epochs 20 --max_batches 200 --seed 0 \
  --count_thr 1.0 \
  --nav_control controlnet --nav_control_scale 1.0 \
  --offroad_weight 0.1 --offroad_field dist --offroad_dist_sigma 3.0 \
  --offroad_samples_per_segment 16 --offroad_agg lse --offroad_lse_beta 10 \
  --log_every 100

止损线（Macro 可行域学习）：
- 风险 1：`offroad_pen_w`（lse）在早期爆炸（例如 >10）或明显震荡不下降 → 立刻停止，改架构（增强条件编码/提高 patch 可辨识度）。
- 风险 2：`offroad_pen_w` 降到很低但 `collision_rate_any` 仍 >50% → proxy gap；可做一次增密（提高 `offroad_samples_per_segment`）验证，仍不行则改架构。
- 风险 3：训练到 20 epoch 仍无法把 `collision_rate_any` 压到 <10% → 认为达到表达极限，改架构。
```

3) 采样 Macro→Skeleton（不接 micro，先验证 G1/G2；建议在 detour-hard 子集上跑）：
```bash
python -m src.evaluation.dump_macro_diffusion_samples \
  --exp_name phys_macroZdiff_skeleton_detourhard \
  --macro_checkpoint data/experiments/macro_zdiff_k2_count_current/last.pt \
  --data_path $DATA --nav_file $NAV --split test \
  --obs_len 8 --pred_len 12 \
  --patch_size 32 --nav_patch_channel2 count \
  --k_samples 20 \
  --save_samples 400 --max_batches 13 --seed 0 \
  --windows_npz data/experiments/gt_passenger_dt30_test/test_detour_hard_top10.npz

# 若 G1（碰撞率）不通过：启用 Feasible Gate（accept/reject）在采样端做硬可行性约束，
# 等价于从 p(z|cond, feasible) 的截断分布采样；不依赖 KDE/典型性过滤。
python -m src.evaluation.dump_macro_diffusion_samples \
  --exp_name phys_macroZdiff_skeleton_detourhard_feasible \
  --macro_checkpoint data/experiments/macro_zdiff_k2_count_current/last.pt \
  --data_path $DATA --nav_file $NAV --split test \
  --obs_len 8 --pred_len 12 \
  --patch_size 32 --nav_patch_channel2 count \
  --k_samples 20 \
  --feasible_gate --gate_count_thr 1.0 --gate_sample_step 0.5 --gate_oversample 3 \
  --save_samples 400 --max_batches 13 --seed 0 \
  --windows_npz data/experiments/gt_passenger_dt30_test/test_detour_hard_top10.npz

# G1：可行性（碰撞/越界）门槛：collision_rate_any <= 10%
python -m src.evaluation.macro_waypoint_gate \
  --samples_npz data/experiments/phys_macroZdiff_skeleton_detourhard/samples.npz \
  --nav_file $NAV --count_thr 1.0 \
  --out_json data/experiments/phys_macroZdiff_skeleton_detourhard/macro_waypoint_gate.json

# G2：拓扑（建议 detour_pct=100，把整个子集当 detour 来检验稳定性）
python -m src.evaluation.detour_validity \
  --inputs "MacroSkel:data/experiments/phys_macroZdiff_skeleton_detourhard/samples.npz" \
  --use_all_k --k_max 10 \
  --ds 0.5 --lags 1 2 4 8 --offset_fracs 0 0.25 0.5 0.75 \
  --detour_pct 100 --bootstrap 200 --noise_splits 200 \
  --out_json data/experiments/phys_macroZdiff_skeleton_detourhard/detour_validity.json

python -m src.visualization.plot_physical_stats \
  --inputs "MacroSkel:data/experiments/phys_macroZdiff_skeleton_detourhard/samples.npz" \
  --use_all_k --k_max 10 \
  --turn_min_speed 0.1 --dcv_speed_pctl 99.5 --dcv_accel_pctl 99.5 \
  --save_metrics --output_dir essay/figures/physical_stats \
  --stem fig_physical_stats_macroSkel_zdiff_k2
```

4) 通过后再接 micro（DetRes executor）做闭环（G3）：
```bash
python -m src.evaluation.dump_macro_diffusion_samples \
  --exp_name phys_macroZdiff_detres_detourhard \
  --macro_checkpoint data/experiments/macro_zdiff_k2_count_current/last.pt \
  --micro_checkpoint data/experiments/phys_oracleWP_detres_k2/last.pt \
  --data_path $DATA --split test \
  --obs_len 8 --pred_len 12 \
  --batch_size 32 --num_workers 8 \
  --patch_size 32 --nav_patch_channel2 count \
  --k_samples 20 \
  --save_samples 400 --max_batches 13 --seed 0 \
  --windows_npz data/experiments/gt_passenger_dt30_test/test_detour_hard_top10.npz
```

然后用与 Go/No-Go 3 相同的两套指标（`detour_validity` + `plot_physical_stats`）评估：
```bash
python -m src.evaluation.detour_validity \
  --inputs "Macro+DetRes:data/experiments/phys_macroZdiff_detres_detourhard/samples.npz" \
  --use_all_k --k_max 10 \
  --ds 0.5 --lags 1 2 4 8 --offset_fracs 0 0.25 0.5 0.75 \
  --detour_pct 100 --bootstrap 200 --noise_splits 200 \
  --out_json data/experiments/phys_macroZdiff_detres_detourhard/detour_validity.json

python -m src.visualization.plot_physical_stats \
  --inputs "Macro+DetRes:data/experiments/phys_macroZdiff_detres_detourhard/samples.npz" \
  --use_all_k --k_max 10 \
  --turn_min_speed 0.1 --dcv_speed_pctl 99.5 --dcv_accel_pctl 99.5 \
  --save_metrics --output_dir essay/figures/physical_stats \
  --stem fig_physical_stats_macroDetRes_zdiff_k2
```

**Go/No-Go（Phase 3）建议口径**：
- detour 子集上，`MacroSkel` 的 `JSD_turn@4/8` 与 `JSD_max_dev_ratio` 相比 `StraightK0`（或 start→end 的朴素骨架）必须显著下降；
- `Macro+DetRes` 不得破坏 `MacroSkel` 的 coarse 指标，同时 `JSD_Speed/JSD_Accel` 应显著优于 `MacroSkel`；
- 若上述不成立：更可能是信息缺失（需要 weak-map/graph），而不是立刻换更重的 macro 生成器。

---

## 6) Phase 4：端到端组合（小规模闭环 + 可归因）

每个 case 必须同时产出/保存三条轨迹（同条件）：
1) skeleton-only
2) skeleton + deterministic residual
3) skeleton + diffusion residual

并在同一套 Phase 0 指标（overall + detour subset）下出报表，锁死归因边界：
- waypoint（拓扑决策）
- skeleton（几何生成器）
- micro（纹理/可行性执行）

---

## 7) 最小实验序列（按“最省卡、最强止损”排序）

1) Phase 0：定义 detour subset + 空间尺度 turn + max_dev_ratio/len_ratio + CI + noise floor  
2) Phase 1：Oracle waypoints → skeleton-only 是否保住 coarse detour（含 0 waypoint 对照）  
3) Phase 2：skeleton-only vs deterministic residual vs diffusion residual（三对照）  
4) Phase 3：macro 先只评 skeleton 指标（别急着接 micro），通过后再组合  
5) Phase 4：小 K、小子集闭环（可归因产物齐全）

# Residual Diffusion（v1.1）：用 Deterministic Baseline 做 Prior

本说明用于解决 v1.0 中观察到的 **Mean-Reversion Shrinkage**：纯 diffusion / physics diffusion 的典型速度偏小（macro: MSD/Rog 偏低）。

> **范围声明**：本文件讨论的是 Phase B（dt30，窗口级）里的 shrinkage 结构性修复（prior+residual）。  
> 若你当前在跑 trip-level 的分层路线（Macro Hard Support + AR + DetRes）与后续 OSM/拓扑/语义路线，请转看：  
> - `docs/archive/legacy_shenzhen/PHASE_C_RESULTS.md`  
> - `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`

核心思路（KISS）：

- 冻结一个确定性模型（`SeqBaseline`）作为 **prior**：$\hat{v}_{prior} = f_{base}(\text{obs}, \text{cond})$
- diffusion/physics 不再直接建模 $v$，而是建模 **残差**：
  $$v = v_{prior} + v_{residual}$$

代码支持：

- `src/training/train_diffusion.py`：新增 `--prior_checkpoint`，训练时自动用 `vel_full - vel_prior` 作为 target（diff loss 训练 residual；macro loss 若启用则在 full trajectory 上计算）
- `src/training/evaluate.py`：新增 `--prior_checkpoint`，评估时自动将采样到的 residual 加回 prior

> 重要建议（KISS）：Residual 模式的初衷是让 prior 承担宏观幅度，因此 **默认不要再叠加 macro loss**（`--lambda_rog 0`）。  
> 若一定要加，先用很小权重并做门控/归一化，否则容易“变量耦合”导致定位困难（详见 pitfalls 讨论）。

---

## 0) 前置：确保 dt30 数据与 baseline prior 存在

```bash
DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
NAV=data/processed_dt30/nav_field.npz
PRIOR=data/experiments/baseline_b_dt30/last.pt
```

若 `PRIOR` 不存在，先训练 baseline（dt30）：

```bash
python -m src.training.train_baseline \
  --exp_name baseline_b_dt30 \
  --data_path ${DATA} \
  --split train \
  --obs_len 8 --pred_len 12 \
  --hidden_dim 128 \
  --batch_size 1024 \
  --epochs 50 \
  --lr 1e-3 \
  --num_workers 16 \
  --seed 0
```

---

## 1) Residual Diffusion（Data-only）

### 1.1 训练（full run 示例）

```bash
python -m src.training.train_diffusion \
  --model_type diffusion \
  --data_path ${DATA} \
  --split train \
  --exp_name diff_dt30_residual_priorB_h128_b2048_lr1e-3_e100_s0 \
  --prior_checkpoint ${PRIOR} \
  --obs_len 8 --pred_len 12 \
  --hidden_dim 128 \
  --batch_size 2048 \
  --lr 1e-3 \
  --epochs 100 \
  --num_workers 16 \
  --seed 0
```

### 1.2 评估（test，K=20）

```bash
python -m src.training.evaluate \
  --exp_name diff_dt30_residual_priorB_eval_test \
  --model_type diffusion \
  --data_path ${DATA} \
  --checkpoint data/experiments/diff_dt30_residual_priorB_h128_b2048_lr1e-3_e100_s0/last.pt \
  --prior_checkpoint ${PRIOR} \
  --split test \
  --obs_len 8 --pred_len 12 \
  --batch_size 64 \
  --num_workers 8 \
  --num_samples_per_condition 20 \
  --diff_steps 100 \
  --save_samples 200 \
  --seed 0
```

---

## 2) Residual Physics Diffusion（nav_field conditioning）

### 2.1 训练

```bash
python -m src.training.train_diffusion \
  --model_type physics \
  --data_path ${DATA} \
  --nav_file ${NAV} \
  --split train \
  --exp_name phys_dt30_residual_priorB_h128_b2048_lr1e-3_e100_s0 \
  --prior_checkpoint ${PRIOR} \
  --obs_len 8 --pred_len 12 \
  --patch_size 32 \
  --hidden_dim 128 \
  --batch_size 2048 \
  --lr 1e-3 \
  --epochs 100 \
  --num_workers 16 \
  --seed 0
```

### 2.2 评估（test，K=20）

```bash
python -m src.training.evaluate \
  --exp_name phys_dt30_residual_priorB_eval_test \
  --model_type physics \
  --data_path ${DATA} \
  --nav_file ${NAV} \
  --checkpoint data/experiments/phys_dt30_residual_priorB_h128_b2048_lr1e-3_e100_s0/last.pt \
  --prior_checkpoint ${PRIOR} \
  --split test \
  --obs_len 8 --pred_len 12 \
  --patch_size 32 \
  --batch_size 64 \
  --num_workers 8 \
  --num_samples_per_condition 20 \
  --diff_steps 100 \
  --save_samples 200 \
  --seed 0
```

---

## 3) 快速自检（建议看这些字段）

`data/experiments/<eval_exp>/metrics.json` 里建议优先检查：

- `pred_speed_mean / gt_speed_mean`：是否从 v1.0 的 ~0.6 拉回到接近 1（至少不低于 baseline）
- `Rog / GT_Rog`、`MSD_10 / GT_MSD_10`：宏观幅度是否至少达到 baseline 水平
- `ADE_best / FDE_best`：Residual 后是否还能保持/提升 best-of-K（多样性）

如果 residual 模式下宏观仍显著偏小，优先排查：

- `PRIOR` 是否是 dt30 对应的 baseline（不要混用非 dt-fixed 的 prior）
- 是否忘记在评估时加 `--prior_checkpoint`（否则你评估的是 residual 本身，会极小）

---

## 4) 已验证事实（fast eval，供快速决策）

> 注意：以下均为 *fast/subset* 评估（用于确认方向是否正确），不是最终 paper table。

### 4.1 Data-only Residual（test, K=10）

- 指标文件：`data/experiments/diff_residual_test_fast/metrics.json`
- 观察：
  - `pred_speed_mean ≈ gt_speed_mean`（速度比约 1.016）→ **宏观幅度基本恢复**
  - `Rog ≈ GT_Rog`（比约 1.004）→ **RoG 恢复**
  - `MSD_10` 略低于 GT（比约 0.935）→ 仍可能存在一定方向抵消，但已明显优于 v1.0 的 ~0.6×

### 4.2 Physics Residual（test, K=10）

- 指标文件：`data/experiments/phys_residual_test_fast/metrics.json`
- 观察：
  - micro 更强（ADE/FDE/best-of-K 优于 data-only residual）
  - 宏观仍略保守：`Rog/GT_Rog ≈ 0.945`，`MSD_10/GT_MSD_10 ≈ 0.864`
  - 解释：nav_field 像“local mean-flow tether”，会让模型更稳但更保守；Residual 能显著缓解收缩，但 physics 侧仍有剩余偏差需要进一步隔离变量排查。

---

## 5) 失败的 ablation：`nav_patch_channel2=zeros`（direction-only）

动机：尝试只给方向信息，把速度幅度完全交给 residual/prior。

实现方式：评估/训练时使用 `--nav_patch_channel2 zeros`（将 nav_patch 的第 2 通道置零，只保留方向场）。

结果（val, K=10）：

- 未微调：`data/experiments/phys_residual_val_fast_dironly/metrics.json`
  - `Rog/GT_Rog ≈ 0.818`，`MSD_10/GT_MSD_10 ≈ 0.690`（明显变差）
- 微调后：`data/experiments/phys_residual_val_fast_dironly_ft/metrics.json`
  - `Rog/GT_Rog ≈ 0.893`，`MSD_10/GT_MSD_10 ≈ 0.803`（有所回升，但仍落后于当前 physics residual）

结论：direction-only 会引入分布偏移与更保守的行为，不作为主路线。

---

## 6) 有限收益的 ablation：`nav_emb_scale`（影响快速饱和）

在 `val, K=1, max_batches=50` 的快速扫参中，`nav_emb_scale` 从 0.5→1.25 呈单调改善但很快饱和：

| `nav_emb_scale` | Speed Ratio | RoG Ratio | MSD10 Ratio |
|---:|---:|---:|---:|
| 0.5 | 0.885 | 0.847 | 0.706 |
| 0.75 | 0.925 | 0.888 | 0.773 |
| 1.0 | 0.944 | 0.908 | 0.806 |
| 1.25 | 0.948 | 0.911 | 0.813 |

结论：
- `nav_emb_scale` 不是“主杠杆”，继续扫参性价比很低；
- 推荐默认先用 `nav_emb_scale=1.0`，把时间花在更结构性的变量（prior/残差建模、conditioning 设计、采样策略）上。

---

## 6.1) 新主线：NavGate（learnable gating，替代静态 `nav_emb_scale`）

动机（KISS）：
- 当前 physics residual 的剩余偏差更像 mean-field tether（保守化），而 `nav_emb_scale` 是弱旋钮（1% 量级）。
- 我们希望模型能“按条件自适应地信/不信 nav\_field”，而不是用一个全局常数去调。

实现要点：
- 在 `PhysicsConditionDiffusion` 中加入可学习 gate：`nav_emb *= sigmoid(MLP([obs, cond]))`，gate $\in (0,1)$；
- 直观上 gate 只会“减弱 tether”，不负责把尺度拉大（尺度由 prior 负责）。

训练（对照实验建议先跑 `epochs=20,max_batches=200` 快速证伪）：

```bash
DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
NAV=data/processed_dt30/nav_field.npz
PRIOR=data/experiments/prior_dt30_dispw_clip0.5_5_h128_b1024_lr1e-3_e50_s1/last.pt  # 推荐 prior；若你的目录名不同请替换

python -m src.training.train_diffusion \
  --model_type physics \
  --data_path ${DATA} \
  --nav_file ${NAV} \
  --split train \
  --exp_name phys_residual_navGate_obscond_e20_s0 \
  --prior_checkpoint ${PRIOR} \
  --hidden_dim 128 --batch_size 2048 --lr 1e-3 --epochs 20 --max_batches 200 \
  --num_workers 16 --seed 0 \
  --nav_gate obscond --nav_gate_hidden 32 --nav_gate_dropout 0.0
```

评估（建议 `ds100`；`--nav_gate auto` 会从 checkpoint 自动对齐，无需手填）：

```bash
python -m src.training.evaluate \
  --exp_name phys_residual_navGate_obscond_val_k10_mb200_ds100 \
  --model_type physics \
  --data_path ${DATA} \
  --nav_file ${NAV} \
  --checkpoint data/experiments/phys_residual_navGate_obscond_e20_s0/last.pt \
  --prior_checkpoint ${PRIOR} \
  --split val \
  --batch_size 256 --num_workers 0 --max_batches 200 \
  --num_samples_per_condition 10 --diff_steps 100 \
  --nav_gate auto
```

成功判据（止损线）：
- `RoG/MSD10` 相比 concat 基线回升 `≥0.01`；
- 不出现明显 jitter（`pred_path_len_mean` 不异常暴涨）。

### 6.1.1) 最新 quick 证据（val, ds100, K=10, B=200）：NavGate v0 未达止损线

在相同设置下对比 `concat` vs `NavGate(obscond)`（同一评估口径）：

| 结构 | Speed Ratio | RoG Ratio | MSD10 Ratio | ADE\_best | FDE\_best |
|---|---:|---:|---:|---:|---:|
| Concat\_k10 | 0.9692 | 0.9510 | 0.8777 | 4.14 | 5.53 |
| NavGate\_k10 | 0.9584 | 0.9414 | 0.8600 | 4.11 | 5.50 |

结论（事实）：NavGate 在该轮实验中 **macro 指标略降**（未达到 `≥0.01` 的回升要求），同时 micro 指标略有改善；因此 **不作为 Phase B 窗口级主线修复**，后续若继续探索需先定位 gate 的行为（gate 是否饱和、是否“关掉了”有用的 nav 信号）。

---

## 7) 评估成本控制（强烈建议）

为了避免“跑一晚才知道没效果”，建议统一使用两阶段评估：

1) **粗筛**（确认方向）：`K=1, max_batches=50`
2) **精验**（确认趋势）：`K=10, max_batches=200`
3) **最终**（paper）：`K=20, full test`（或至少 `max_batches=2000`）

并行建议：
- 同一台多 GPU：每张卡跑一个 eval（不同 `exp_name`），用 `python -u ... |& tee` 保留实时进度。
- HDF5 并行不稳时：设置 `HDF5_USE_FILE_LOCKING=FALSE`，必要时 `--num_workers 0`。

---

## 8) 教授 Review 对齐（v1.1 之后的优先级）

我们收到教授的批判性 review 后，当前路线的优先级与结论更加明确：

### A) Prior Quality Check（首要任务）

Residual 的天花板很大程度取决于 prior（deterministic baseline）的质量。若 prior 本身就系统性低位移（L2 平滑效应），residual 需要承担过大的“拉伸”压力，容易出现：

- residual 变成“补丁”而不是不确定性建模；
- 或者宏观恢复受限（MSD10/Rog 上不去）。

因此建议立刻做 displacement-aware 的 deterministic prior（按 GT 位移大小加权 MSE）。

> 代码支持：`src/training/train_baseline.py` 已提供 `--disp_weight {none,tanh,clip}` 等参数（见仓库说明）。

补充（PI/组内共识，KISS）：

- **暂时不换 Transformer prior**：在 Phase B 的窗口级主线里，我们要证明的是 *Residual Framework*（scale vs stochasticity 解耦）与 *physics conditioning* 的作用机理。换成 Transformer 会引入大量新变量（收敛/过拟合/实现差异），导致无法归因。
- **先“修目标函数”而不是“换架构”**：deterministic prior 的主要偏差来自 MSE 的均值回归（mean reversion），优先用 displacement-aware weighting 把宏观尺度抬起来，再让 residual 专注学习随机性。

我们在 dt30 的 smoke→full prior 训练中已经看到非常明确的信号：

- `tanh`（只会压低小位移，权重上限≤1）在 smoke budget 下几乎无收益（宏观 ratio 与未加权 baseline 接近）。
- `clip`（允许 `w>1`，把长位移样本显式放大）在同预算 smoke 下显著抬升宏观尺度，并在 full run（e50）达到稳定的 “>=GT” 先验幅度：
  - test, K=1：speed≈1.01–1.03，Rog≈1.06–1.09，MSD10≈1.03–1.08（seed0/1）

结论：**Prior 侧必须使用允许 `w>1` 的加权策略**，否则 residual 会被迫承担宏观尺度修正，天花板很低。

### B) Nav Field 的更精细交互（避免 mean-field tether）

当前 physics 条件注入是 `nav_emb` 与 `cond` 的拼接（concat）。它能稳定方向，但也可能像“锚链”把生成拉向局部均值。

短期（KISS）建议优先做“低成本、可证伪”的 ablation，而不是一上来改结构：

1) **Direction-only 是否真的有效？**
   - 我们做过 `nav_patch_channel2=zeros`（只保留方向、移除 channel2），结果更保守（见第 5 节），说明“纯方向”在当前实现下并不奏效。
   - 但这不完全等价于“只提供方向不提供速度”的理想设定：`zeros` 同时移除了“置信度/密度”信息，方向场在低 count 区域可能更噪声，反而会加剧保守行为。

2) **更贴近教授建议的版本：Direction + Count（不提供速度幅度）**
   - 数据层面已经支持：`nav_patch_channel2=count`（方向 + log-count 作为置信度），避免用 speed 通道“限速”。
   - 推荐在 residual physics 上做一个对照组（与 `speed` 版本同配置），用 `K=1,B=50` 快速证伪。

3) 若 tether 仍明显，再考虑结构改动（ControlNet/FILM 注入等），避免变量耦合导致定位困难。

阶段性实验结论（dt30, residual physics, prior=disp-aware clip）：

- `nav_patch_channel2=speed` vs `count` 在 `val, ds100, K=10, B=200` 下差异极小：
  - speed：speed≈0.961，Rog≈0.944，ADE_best≈4.14
  - count：speed≈0.958，Rog≈0.942，ADE_best≈4.18

含义：目前的 “tether/保守” 主要不是由 speed 通道单独造成，更像是 **direction mean-field + concat 注入方式** 的整体效应。

补充：`nav_emb_scale` 在 `val, ds100, K=5, B=100` 的快速对照里同样表现为“弱旋钮”：

- `nav_emb_scale=0.75`：speed≈0.991，Rog≈0.967，MSD10≈0.910，ADE_best≈4.66
- `nav_emb_scale=1.25`：speed≈0.985，Rog≈0.962，MSD10≈0.909，ADE_best≈4.61

差异在 1% 量级，优先级明显低于 “prior 质量” 与 “residual decomposition”。

另一个重要观察（收敛性排雷）：在同一 residual physics 配置下，**继续训练会让 micro 变好但 macro 变差**（典型的 safe-play / local minimum）：

- `e20 (val, ds100, K=10, B=200)`: speed≈0.961，Rog≈0.944，ADE_best≈4.14
- `e40 (val, ds100, K=10, B=200)`: speed≈0.953，Rog≈0.938，ADE_best≈3.94

含义：如果目标是 “Valid Simulation”（宏观真实），不能只追 micro；需要引入训练级低频约束（例如门控的 multi-EPE macro loss）来打破保守陷阱。

### D) Macro-FT（低成本验证）结论：**无效（触发止损）**

在 `e40` checkpoint 上做 5-epoch 的 macro fine-tune（`multi_epe + t<thr(50) + exp(t) + batch_relative`），并行尝试 `λ=0.005/0.01`：

| 设置 | Speed Ratio | RoG Ratio | MSD10 Ratio | ADE_best | FDE_best |
|---|---:|---:|---:|---:|---:|
| e40 ref | 0.9525 | 0.9376 | 0.8574 | 3.94 | 5.34 |
| λ=0.005 | 0.9558 | 0.9363 | 0.8560 | 3.90 | 5.25 |
| λ=0.01 | 0.9569 | 0.9380 | 0.8588 | 3.89 | 5.23 |

结论：
- macro（RoG/MSD10）几乎不动（<0.01），不满足止损线（RoG 回升 ≥0.01）。
- micro 继续变好（ADE/FDE 下降）→ 说明模型仍在向“更保守/更靠均值”的局部最优滑动。

因此：**macro fine-tune 级别难以扭转已收敛的保守分布**。下一步若仍坚持 macro 路线，应优先考虑 **从头训练就引入 macro loss（而不是续训）**，或转向更结构化的 conditioning 注入（P2）。

### C) 采样效率（P2：在保真后再做）

`diff_steps=50` 会产生明显分布漂移，不作为严谨结论依据。当前优先级是 validity（`diff_steps=100`）。
若面向 Digital Twin 落地需要加速，建议后续讨论 distillation/flow matching（P2）。

---

## 9) v1.2：对 diff_loss 做位移加权（Displacement-aware Diff Loss Weighting）

动机（First Principles）：
- residual/physics 在收敛后常见“**微观变好、宏观变差**”的 safe-play 局部最优：模型更愿意待在均值附近以降低点对点误差；
- 这类偏差往往来自 **目标函数被低位移窗口主导**（low-displacement dominance），而不是 sampling temperature 或简单 macro-loss 能稳定解决的。

做法（KISS）：
- 保持 diffusion 的 timestep 采样均匀不变；
- 只对 **diffusion 的 diff_loss** 做 per-sample 权重：
  $$w_i = \mathrm{clip}\Big(\frac{\|\Delta x^{gt}_i\|}{\mathbb{E}[\|\Delta x^{gt}\|]},\; w_{min},\; w_{max}\Big),\;\; w_{max}>1$$
- 其中 $\Delta x^{gt}$ 在实现中由 **反归一化后的 GT 速度**求和得到（单位：grid cells / step），保证语义一致。

对应开关（训练脚本：`src/training/train_diffusion.py`）：
- `--diff_disp_weight clip`
- `--diff_disp_clip_min 0.5`
- `--diff_disp_clip_max 5.0`

### 7.1 建议的最小对照（先快速证伪，再全量）

以 physics residual 为例（推荐先 `epochs=20,max_batches=200` 快速筛）：

```bash
DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
NAV=data/processed_dt30/nav_field.npz
PRIOR=data/experiments/baseline_b_dt30/last.pt

# 对照 A：不加权（baseline）
python -m src.training.train_diffusion \
  --model_type physics --data_path ${DATA} --nav_file ${NAV} --split train \
  --exp_name phys_residual_v12_ref_e20_s0 \
  --prior_checkpoint ${PRIOR} \
  --hidden_dim 128 --batch_size 2048 --lr 1e-3 --epochs 20 --max_batches 200 \
  --num_workers 16 --seed 0 \
  --lambda_rog 0

# 对照 B：v1.2 diff_loss 位移加权
python -m src.training.train_diffusion \
  --model_type physics --data_path ${DATA} --nav_file ${NAV} --split train \
  --exp_name phys_residual_v12_dispw_clip_e20_s0 \
  --prior_checkpoint ${PRIOR} \
  --hidden_dim 128 --batch_size 2048 --lr 1e-3 --epochs 20 --max_batches 200 \
  --num_workers 16 --seed 0 \
  --diff_disp_weight clip --diff_disp_clip_min 0.5 --diff_disp_clip_max 5.0 \
  --lambda_rog 0
```

评估建议（两阶段，控制时间成本）：
1) `K=1,max_batches=50,ds100` 先看 macro ratio 是否回升（止损线：RoG/MSD10 回升 ≥ 0.01）
2) `K=10,max_batches=200,ds100` 再看 best-of-K

> 注意：如果 `speed_ratio > 1.1` 且 `ADE_best` 明显变差，优先把 `--diff_disp_clip_max` 从 5.0 降到 3.0（避免过度放大长位移样本导致 overshoot）。

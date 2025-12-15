# Phase B 结果解读（Paper Strict, dt-fixed=30s）

> **用途**：论文撰写素材的“严格版结果分析 + 证据链目录”，用于回答：  
> 1) dt-fixed（30s）下 pipeline 是否严格可复现、无泄漏？  
> 2) Data-only Diffusion/Physics Diffusion 在 **dt 语义明确** 的条件下，微观/宏观指标是否仍成立？  
>
> **重要边界**：本文件只基于仓库内已有的产物做事实总结。当前 Phase B 仍缺少 Diffusion/Physics 的 **320 条 condition quick eval** 与 **全量 test eval**，因此任何结论都必须标注为 *preliminary*（详见第 6 节）。

---

## 1. Phase B 的目标与主线（为什么必须做）

Phase B 的主线很简单：把 Phase A 的“step-based 趋势验证”升级为 **物理时间语义明确** 的版本，避免审稿人质疑：

- **MSD 标度律**的横轴 $\Delta t$ 含义不清；
- **nav_field 的 speed**（步位移模长）在不同采样间隔下不可比；
- **宏观约束/宏观解释**没有物理基础；
- 结果对数据集/采样间隔不具可复现迁移性。

因此 Phase B 规定：**必须重采样到固定 `dt_fixed=30s`**，并且所有 strict 数据产物（`data_stats.json/nav_field.npz`）必须 **train-only** 且记录来源 hash（无泄漏合同）。

---

## 2. 数据与 strict 合同（dt30 数据集）

### 2.1 产物路径（单一真相源）

- dt30 轨迹与 split：`data/processed_dt30/trajectories/shenzhen_trajectories.h5`、`data/processed_dt30/splits/*.npy`
- 重采样合同：`data/processed_dt30/resample_meta.json`
- train-only 统计量：`data/processed_dt30/data_stats.json`
- train-only 导航场：`data/processed_dt30/nav_field.npz`

### 2.2 重采样输出统计（来自 `resample_meta.json`）

- 输入轨迹数：154285
- 输出轨迹数：139634（丢弃 14651）
  - `gap_too_large`：10897（超过 `max_gap=300s`）
  - `too_short_duration`：3754（不足 `min_length=10`）
- split（trip-id）输出：
  - train：97222，val：21103，test：21309

### 2.3 strict sanity check（dt-fixed + nav 对齐）

`python -m src.utils.sanity_check --data_path data/processed_dt30 --strict --expected_dt 30 --dt_require_constant` 的关键结论：

- dt 全量检查：`min=max=30`，无 0/负/不一致间隔（跨轨迹边界已剔除）
- nav_field 对齐（`min_count=10` 的格子）：
  - `mean_cos=0.255`，`mean|cos|=0.785`，`neg_ratio=0.362`

> 解释：道路双向流会拉低 `mean_cos`，因此 `mean|cos|` 更适合作为一致性诊断。

---

## 3. 模型与训练配置（Phase B, dt30）

### 3.1 三类模型（与 Phase A 一致）

- Baseline：`src/models/seq/seq_baseline.py`
- Data-only Diffusion：`src/models/diffusion/diffusion_model.py`
- Physics Diffusion（nav_patch 条件）：`src/models/physics/physics_condition_diffusion.py`

### 3.2 训练产物与超参（来自 checkpoint config）

权重（本地）：

- Baseline：`data/experiments/baseline_b_dt30/last.pt`
- Diffusion：`data/experiments/diff_b_dt30/last.pt`
- Physics：`data/experiments/physics_b_dt30/last.pt`

关键超参（Diffusion/Physics）：

- `obs_len=8`，`pred_len=12`
- `hidden_dim=64`
- `diff_steps=100`
- `patch_size=32`（Physics）
- `epochs=50`，`batch_size=2048`，`lr=1e-3`

Baseline（从权重推断）：

- `hidden_dim=128`

---

## 4. Phase B 当前已有评估结果（preliminary）

### 4.1 评估产物路径

- Baseline quick（320 条 condition）：`data/experiments/baseline_b_dt30_eval_quick/metrics.json`
- b1 子集（32 条 condition，对齐同一批样本的最小闭环）：
  - Baseline：`data/experiments/baseline_b_dt30_eval_b1/metrics.json`
  - Diffusion：`data/experiments/diff_b_dt30_eval_b1/metrics.json`
  - Physics：`data/experiments/physics_b_dt30_eval_b1/metrics.json`

### 4.2 微观指标（b1 子集，越小越好）

| 模型 | K | ADE_mean | FDE_mean | Fréchet_mean | DTW_mean |
|---|---:|---:|---:|---:|---:|
| Baseline | 1 | 5.012 | 6.357 | 6.885 | 44.810 |
| Diffusion | 20 | 7.945 | 13.907 | 14.313 | 84.483 |
| Physics | 20 | 7.550 | 12.814 | 13.238 | 79.008 |

**现象（必须正视）**：

- 在该 32 条子集上，Diffusion/Physics 的微观误差 **显著劣于 Baseline**；Physics 相对 Diffusion 有一致的小幅改善，但不足以追平 Baseline。

> 这与 Phase A（Physics 在微观指标上优于 Baseline）的趋势不同，提示 Phase B 的训练/采样/容量需要进一步排查与调参（见第 6 节）。

### 4.3 宏观指标与 GT 对照（b1 子集）

| 指标 | GT | Baseline | Diffusion | Physics |
|---|---:|---:|---:|---:|
| MSD_1 | 5.917 | 2.100 | 2.314 | 3.365 |
| MSD_5 | 108.282 | 50.899 | 32.701 | 47.083 |
| MSD_10 | 352.869 | 198.284 | 96.700 | 134.381 |
| Rog | 6.455 | 4.683 | 3.224 | 4.035 |

**宏观解读（preliminary）**：

- 三个模型在该子集上整体偏“收缩”（MSD/Rog < GT），其中 Diffusion 最明显。
- Physics 相对 Diffusion 的 MSD/Rog 更大（更接近 GT），说明 nav_field 条件对“运动幅度”有正向拉回作用；但仍未达到 GT。

### 4.4 Baseline quick（320 条 condition）补充（仅 Baseline）

`baseline_b_dt30_eval_quick` 显示（320 条子集）：

- 微观：ADE_mean=5.466，FDE_mean=8.850
- 宏观对照：Rog=5.492 vs GT_Rog=5.247（相对误差约 4.7%），但 MSD_1/5/10 仍明显偏低（步位移方差不足）。

> 注意：b1 与 quick 的子集不同，不能直接做跨表格结论；这里只用于展示“Baseline 在不同子集上的稳定性区间”。

---

## 5. 可视化（论文级图件）

Phase B 的“论文级图件”建议与 Phase A 保持同一风格，至少包括：

1. 微观指标对比（ADE/FDE/Fréchet/DTW，mean + best-of-K）
2. MSD 曲线（log-log，横轴 $\tau=k\\cdot 30s$，含 GT 对照与幂律指数）
3. 同一条件下的轨迹叠图（GT + Baseline + Diffusion + Physics）
4. ADE/FDE 的 CDF（基于保存样本）
5. Rog 的分布（箱线图/小提琴图）

对应脚本见：`src/visualization/plot_phase_b_report.py`（与 Phase A 脚本保持同风格）。

运行命令（默认读取 b1 子集的三组评估目录，并输出 PDF+PNG）：

```bash
MPLCONFIGDIR=/tmp/mplconfig \
  ~/miniconda3/envs/emotion/bin/python -m src.visualization.plot_phase_b_report \
  --out_dir data/experiments/phase_b_report/figures
```

---

## 6. 下一步必须补齐的“论文版闭环”（行动清单）

> 目标：让 Phase B 的结论“方法论严谨 + 实验可复现 + 结论站得住”。

### 6.1 补齐最关键的缺口：Phase B 的 quick eval（320 条）与更大规模评估

必须在 **同一套 dt30 产物** 上完成：

- Diffusion quick（320 条，`K=20, diff_steps=100`）
- Physics quick（320 条，`K=20, diff_steps=100`）
- （建议）更稳的中等规模：`max_batches=200`（或全量）
- 多随机种子（至少 3 个）验证趋势是否稳定

### 6.2 如果 Phase B 仍然“Baseline > Diffusion/Physics”（需要立刻排雷）

优先排查顺序（KISS）：

1. **采样是否与训练一致**：`diff_steps`、`obs_len/pred_len`、`hidden_dim` 是否对齐 checkpoint（evaluate.py 已自动对齐 hidden_dim）。
2. **容量/批大小是否导致欠拟合**：`hidden_dim=64 + batch_size=2048` 可能偏欠拟合（建议做 `hidden_dim=128`、`batch_size=512/1024` 的对照）。
3. **归一化统计量是否正确**：确认 dt30 的 `data_stats.json` 来自 train split，且训练/评估都读取 dt30 目录下的 stats（目前 sanity check 已 PASS）。
4. **OD 条件是否有效**：检查 cond 编码是否被模型使用（可做 ablation：只用 obs vs obs+OD）。

---

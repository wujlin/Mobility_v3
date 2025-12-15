# Phase B 结果解读（Paper Strict, dt-fixed=30s）

> **用途**：论文撰写素材的“严格版结果分析 + 证据链目录”，用于回答：  
> 1) dt-fixed（30s）下 pipeline 是否严格可复现、无泄漏？  
> 2) Data-only Diffusion/Physics Diffusion 在 **dt 语义明确** 的条件下，微观/宏观指标是否仍成立？  
>
> **重要边界**：本文件只基于仓库内已有的产物做事实总结。当前 Phase B 已补齐 **320 条 condition quick eval**，但仍缺少 **更大规模/全量 test eval + 多随机种子复现**，因此任何结论都必须标注为 *preliminary*（详见第 6 节）。

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

- Quick Eval（320 条 condition，K=20）：  
  - Baseline：`data/experiments/baseline_b_dt30_eval_quick/metrics.json`（+ `samples.npz`）  
  - Diffusion：`data/experiments/diff_b_dt30_eval_quick/metrics.json`（+ `samples.npz`）  
  - Physics：`data/experiments/physics_b_dt30_eval_quick/metrics.json`（+ `samples.npz`）
- b1 子集（32 条 condition，debug 用的最小闭环，避免跑太久）：  
  - `data/experiments/*_b_dt30_eval_b1/`

### 4.2 微观指标（Quick Eval：320 条 condition，越小越好）

> 说明：Diffusion/Physics 的 `*_std` 是 **每个 condition 内 K 次采样的波动**；`*_best` 为 best-of-K（oracle 上界，用于衡量覆盖潜力）。

| 模型 | K | ADE_mean | ADE_std | ADE_best | FDE_mean | FDE_std | FDE_best | Fréchet_mean | Fréchet_std | Fréchet_best | DTW_mean | DTW_std | DTW_best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1 | 5.467 | 0.000 | 5.467 | 8.855 | 0.000 | 8.855 | 9.234 | 0.000 | 9.234 | 54.496 | 0.000 | 54.496 |
| Diffusion | 20 | 6.741 | 2.651 | 2.836 | 11.636 | 5.025 | 3.950 | 11.838 | 4.956 | 4.433 | 72.064 | 33.670 | 26.194 |
| Physics | 20 | 6.744 | 2.926 | 2.510 | 11.644 | 5.556 | 3.419 | 11.863 | 5.464 | 4.028 | 71.981 | 36.824 | 22.623 |

**现象（必须正视）**：

- 在该 quick 子集上，Diffusion/Physics 的 **mean 口径**微观误差显著劣于 Baseline（ADE/FDE/Fréchet/DTW 均 ↑约 23–32%）。  
- Physics 的 **mean** 与 Diffusion 基本持平，但 **best-of-K** 明显更好（四个指标均比 Diffusion 的 best-of-K 低约 9–14%），说明 nav_field 条件更像是在提升“覆盖潜力/上界”，而不是提升“典型样本质量”。

> 这与 Phase A（Physics 在 mean 口径上优于 Baseline）的趋势不同：Phase B(dt30) 下 Baseline 很强，而 Diffusion/Physics 出现明显的“低位移幅度/偏收缩”问题（见 4.3）。

### 4.3 宏观指标与 GT 对照（Quick Eval：同 320 条 condition）

| 指标 | GT | Baseline | Diffusion | Physics |
|---|---:|---:|---:|---:|
| MSD_1 | 5.345 | 3.322 | 2.136 | 2.578 |
| MSD_5 | 103.678 | 81.603 | 32.945 | 38.700 |
| MSD_10 | 349.740 | 304.099 | 100.948 | 116.869 |
| Rog | 5.247 | 5.494 | 3.056 | 3.435 |

**宏观解读（preliminary）**：

- Baseline 的 Rog 与 GT 很接近（约 4.7% 相对误差），MSD 也相对更接近（尤其 MSD_10）。  
- Diffusion/Physics 的 MSD 与 Rog 明显偏低，表现为生成轨迹整体“收缩/走不动”，导致微观误差（尤其 FDE）偏大。  
- Physics 相对 Diffusion 的 MSD/Rog 更大（更接近 GT），说明 nav_field 条件确实在“拉回运动幅度”，但目前仍不足以达到可论文结论的水平。

---

## 5. 可视化（论文级图件）

Phase B 的“论文级图件”建议与 Phase A 保持同一风格，至少包括：

1. 微观指标对比（ADE/FDE/Fréchet/DTW，mean + best-of-K）
2. MSD 曲线（log-log，横轴 $\tau=k\\cdot 30s$，含 GT 对照与幂律指数）
3. 同一条件下的轨迹叠图（GT + Baseline + Diffusion + Physics）
4. ADE/FDE 的 CDF（基于保存样本）
5. Rog 的分布（箱线图/小提琴图）

对应脚本见：`src/visualization/plot_phase_b_report.py`（与 Phase A 脚本保持同风格）。

运行命令（读取 quick 320 的三组评估目录，并输出 PDF+PNG）：

```bash
MPLCONFIGDIR=/tmp/mplconfig \
  ~/miniconda3/envs/emotion/bin/python -m src.visualization.plot_phase_b_report \
  --baseline_dir data/experiments/baseline_b_dt30_eval_quick \
  --diff_dir data/experiments/diff_b_dt30_eval_quick \
  --physics_dir data/experiments/physics_b_dt30_eval_quick \
  --out_dir data/experiments/phase_b_report/figures_quick
```

---

## 6. 下一步必须补齐的“论文版闭环”（行动清单）

> 目标：让 Phase B 的结论“方法论严谨 + 实验可复现 + 结论站得住”。

### 6.1 补齐最关键的缺口：Phase B 的 quick eval（320 条）与更大规模评估

必须在 **同一套 dt30 产物** 上完成：

- （已完成）Diffusion/Physics quick（320 条，`K=20, diff_steps=100`）
- （下一步）更稳的中等规模：`max_batches=200`（或全量）
- 多随机种子（至少 3 个）验证趋势是否稳定

### 6.2 如果 Phase B 仍然“Baseline > Diffusion/Physics”（需要立刻排雷）

优先排查顺序（KISS）：

1. **采样是否与训练一致**：`diff_steps`、`obs_len/pred_len`、`hidden_dim` 是否对齐 checkpoint（evaluate.py 已自动对齐 hidden_dim）。
2. **容量/批大小是否导致欠拟合**：`hidden_dim=64 + batch_size=2048` 可能偏欠拟合（建议做 `hidden_dim=128`、`batch_size=512/1024` 的对照）。
3. **归一化统计量是否正确**：确认 dt30 的 `data_stats.json` 来自 train split，且训练/评估都读取 dt30 目录下的 stats（目前 sanity check 已 PASS）。
4. **OD 条件是否有效**：检查 cond 编码是否被模型使用（可做 ablation：只用 obs vs obs+OD）。

---

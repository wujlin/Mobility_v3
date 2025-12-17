# Phase A 结果解读（Fast Validation, step-based）

> **用途**：作为论文撰写素材的“阶段性结果分析 + 证据链目录”，用于回答：  
> 1) Pipeline 是否跑通且评估闭环可信？ 2) 生成模型相对预测 baseline 是否有收益？ 3) 物理条件（nav_field）是否带来可观改进？  
>
> **重要边界**：Phase A 不做 `dt_fixed` 重采样，时间轴为 **step**（离散步），不应做严格“物理时间”的结论（详见 `docs/TASK_DEFINITION.md`）。

---

## 1. 实验目的与主线

Phase A 的目标是 **快速验证方向**，而不是“论文版最终结论”：

1. **正确性**：数据产物（split、normalization、nav_field）与代码一致、无泄漏，评估脚本能稳定输出。
2. **趋势**：对比三类模型在同一任务下的表现趋势：
   - Deterministic L2 Regression（SeqBaseline，确定性序列预测）
   - Data-only Diffusion（纯数据生成）
   - Physics Diffusion（引入 nav_field 的物理条件生成）
3. **效应归因**：Physics 相对 Diffusion 的提升是否稳定出现在多个微观指标上（ADE/FDE/Fréchet/DTW）。

---

## 2. 任务与数据设定（Phase A）

### 2.1 任务定义（必须明确）

- **KnownDestination**：推理时已知 trip-level `o,d,t0`，`d` 是合法输入，不算信息泄漏。
- **窗口级未来段生成**：给定 `obs`（历史 H 步），生成未来 `F` 步 `vel`。
- `vel` 语义：**step displacement**（步位移），单位 `grid_cell/step`。

### 2.2 数据与 split

- 数据：深圳出租车轨迹（2011/04/18–26）
- 轨迹数量（trip-level）：154285
- split（trip-id）：
  - train：107999
  - val：23143
  - test：23143

### 2.3 训练与评估配置（本次 Phase A）

- 观测/预测窗口：`obs_len=8`, `pred_len=12`
- 生成采样数：Diffusion/Physics 为 `K=20`；Deterministic L2 为 `K=1`
- 评估集：test split 的窗口样本
- **Quick Eval 子集**：`num_conditions=320`（`batch_size=32, max_batches=10`，即 test 窗口序列的前 320 个）

> 说明：test split 的总窗口数为 294930（`batch_size=32` 时约 9217 个 batch）。Diffusion/Physics 全量评估（K=20, diff_steps=100）会非常耗时，Phase A 先用 quick eval 验证趋势。

---

## 3. Phase A 结果总览（Quick Eval, 320 条 condition）

结果文件（单一真相源）：

- Deterministic L2：`data/experiments/baseline_a_full_eval_quick/metrics.json`
- Diffusion：`data/experiments/diff_a_full_eval_quick/metrics.json`
- Physics：`data/experiments/physics_a_full_eval_quick/metrics.json`

### 3.1 微观指标（越小越好）

> Diffusion/Physics 的 `*_std` 是 **每个 condition 内 K 次采样的波动**（多样性/不确定性刻画），不是“跨样本的标准差”。  
> `*_best` 为 **best-of-K**（oracle upper bound，推理时不可直接使用但可用于衡量分布覆盖潜力）。

| 模型 | K | ADE_mean | ADE_std | ADE_best | FDE_mean | FDE_std | FDE_best | Fréchet_mean | Fréchet_std | Fréchet_best | DTW_mean | DTW_std | DTW_best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic L2 | 1 | 6.419 | 0.000 | 6.419 | 10.936 | 0.000 | 10.936 | 11.154 | 0.000 | 11.154 | 66.556 | 0.000 | 66.556 |
| Diffusion | 20 | 6.015 | 3.708 | 1.662 | 10.420 | 7.033 | 1.589 | 10.655 | 6.961 | 2.288 | 62.040 | 44.876 | 14.385 |
| Physics | 20 | 5.733 | 3.840 | 1.537 | 9.848 | 7.245 | 1.448 | 10.117 | 7.197 | 2.179 | 57.632 | 45.700 | 13.161 |

**关键对比（mean 口径）**：

- Diffusion vs Deterministic L2：ADE ↓6.30%，FDE ↓4.72%，Fréchet ↓4.47%，DTW ↓6.79%。
- Physics vs Diffusion：ADE ↓4.69%，FDE ↓5.49%，Fréchet ↓5.05%，DTW ↓7.11%。
- Physics vs Deterministic L2：ADE ↓10.69%，FDE ↓9.95%，Fréchet ↓9.30%，DTW ↓13.41%。

### 3.2 宏观指标（MSD/Rog）与 GT 对照（同一 320 条 condition）

> 本段 GT 对照是在相同 320 条窗口上，对 ground-truth future positions 计算得到（step-based）。  
> Phase B（dt-fixed）才允许将 lag 映射到真实时间并做更强物理解释。

| 指标 | GT | Deterministic L2 | Diffusion | Physics |
|---|---:|---:|---:|---:|
| MSD_1 | 5.299 | 4.599 | 5.580 | 6.025 |
| MSD_5 | 95.938 | 110.059 | 74.896 | 82.034 |
| MSD_10 | 318.509 | 405.297 | 232.831 | 253.922 |
| Rog | 4.568 | 6.003 | 4.285 | 4.525 |

宏观“对齐度”（相对 GT）：

- Rog 相对误差：Deterministic L2 31.41%，Diffusion 6.19%，Physics **0.94%**。
- MSD 曲线平均相对误差：Deterministic L2 17.24%，Diffusion 20.44%，Physics **14.63%**（本子集上最佳）。

---

## 4. 主要发现（可以写进论文“验证实验/消融”）

### 发现 A：Physics 相对 Data-only Diffusion 的提升是“全指标一致”的

在 ADE/FDE/Fréchet/DTW 的 **mean 口径**上，Physics 均优于 Diffusion，且提升幅度在 4.7%–7.1%。

解释：nav_field 提供了“道路局部方向/经验速度”先验，使扩散生成在多模态采样中更倾向于落在真实轨迹流形附近，减少无意义的漂移与折返。

### 发现 B：best-of-K 显示生成分布具有更强“覆盖潜力”，但需要谨慎表述

- Diffusion/Physics 的 `*_best` 显著优于其 `*_mean`，说明分布中存在更接近 GT 的样本模式。
- 但 best-of-K 依赖 GT 作为 oracle 选择，推理阶段不可直接使用；论文表述应强调它是“上界能力/覆盖度”指标。

### 发现 C：Physics 在宏观结构上更接近 GT（至少在 Rog 上非常明显）

在本 quick 子集上，Physics 的 Rog 与 GT 几乎重合（相对误差 0.94%），而 Deterministic L2 明显过“扩散”、Diffusion 略偏“收缩”。

这与“物理场作为条件”在宏观形态上提供约束是一致的，但 Phase A 仍只适合做趋势判断。

---

## 5. 可视化（子刊级输出）

已提供一键生成脚本：`src/visualization/plot_phase_a_report.py`

默认读取 Phase A quick eval 的三个目录，并输出到：

`data/experiments/phase_a_report/figures/`

生成内容（PDF+PNG）：

1. `fig1_micro_metrics.pdf/png`：四个微观指标的 mean（含 best-of-K 标记）对比
2. `fig2_msd_curve.pdf/png`：MSD 曲线（可选含 GT 对照）与幂律指数（step-based）
3. `fig3_traj_overlay.pdf/png`：同一组样本上 Deterministic L2/Diffusion/Physics 与 GT 轨迹叠图
4. `fig4_error_cdf.pdf/png`：ADE/FDE 的经验 CDF（基于保存样本的分布）
5. `fig5_rog_boxplot.pdf/png`：Rog 的箱线图（GT 与三模型，基于保存样本）

运行命令：

```bash
MPLCONFIGDIR=/tmp/mplconfig \
  ~/miniconda3/envs/emotion/bin/python -m src.visualization.plot_phase_a_report \
  --processed_dir data/processed \
  --split test \
  --obs_len 8 \
  --pred_len 12 \
  --batch_size 32 \
  --max_batches 10 \
  --out_dir data/experiments/phase_a_report/figures
```

> 若你的 Phase A `metrics.json` 还不包含 `GT_msd_curve/GT_Rog`（旧版评估产物），可先用下列命令生成 GT 宏观对照，再传给可视化脚本：
>
> ```bash
> .venv-wsl/bin/python -m src.utils.compute_gt_macro \
>   --processed_dir data/processed \
>   --split test \
>   --obs_len 8 \
>   --pred_len 12 \
>   --batch_size 32 \
>   --max_batches 10 \
>   --out_json data/experiments/phase_a_report/gt_macro.json
>
> MPLCONFIGDIR=/tmp/mplconfig \
>   ~/miniconda3/envs/emotion/bin/python -m src.visualization.plot_phase_a_report \
>   --gt_macro_json data/experiments/phase_a_report/gt_macro.json \
>   --out_dir data/experiments/phase_a_report/figures
> ```

---

## 6. Phase A 的局限与 Phase B 的必要性（写给审稿人的“防误解说明”）

1. **dt 不是常数**：Phase A 的 step 不是固定秒数，MSD 的横轴只能解释为 step lag，不能做严格物理时间标度结论。
2. **quick 子集**：320 条 condition 只能验证趋势；论文结论需在更大规模（例如 `max_batches=200` 或全量）与多随机种子上复现。
3. **best-of-K 是 oracle**：可用于“覆盖潜力”与生成上界，不应被当作推理可达指标。

Phase B 的必要性：固定 `dt_fixed=30s`、重训并在 dt-fixed 上复现实验，才能给出“方法论严谨 + 可复现 + 结论站得住”的论文版证据链。

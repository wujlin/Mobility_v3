# Phase B 结果解读（Paper Strict, dt-fixed=30s）

> **用途**：论文撰写素材的“严格版结果分析 + 证据链目录”，用于回答：  
> 1) dt-fixed（30s）下 pipeline 是否严格可复现、无泄漏？  
> 2) Data-only Diffusion/Physics Diffusion 在 **dt 语义明确** 的条件下，微观/宏观指标是否仍成立？  
>
> **重要边界**：本文件只基于仓库内已有的产物做事实总结。当前 Phase B 已补齐 **320 条 condition quick eval**，但仍缺少 **更大规模/全量 test eval + 多随机种子复现**，因此任何结论都必须标注为 *preliminary*（详见第 6 节）。

---

## 0.1 Phase B 主线（历史）与当前主线（避免“看错版本”）

Phase B 的这份文档讨论的是 **dt30 窗口级（window-level）** 的 shrinkage / macro–micro trade-off。该阶段我们曾重点探索：**Residual（Prior + Residual）+ CFG（Destination Guidance）** 等路线，用于直接对抗 shrinkage 与宏观收缩。

> 重要：后续在 **trip-level 分层诊断（Phase C）** 中，我们把主要瓶颈重新定位为“宏观决策/执行解耦 + 可行域/拓扑/语义信息缺失”，当前主线已转向：
> - `docs/PHASE_C_RESULTS.md`（Hard Support + AR + DetRes 的已验证基线）
> - `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`（OSM 道路先验（软） + 拓扑 + 城市语义 + AR + Diffusion 多模态）

- **子刊级可视化与一键出图口径**：`docs/archive/phase_b/PHASE_B_CFG_VISUALIZATION.md`
  - 对齐子集 `samples.npz` 生成、深圳 geojson 底图叠加、Pareto（cfg 旋钮图）、spaghetti 微观案例图、动画
- **路线图与“哪些坑已止损”**：`docs/archive/phase_b/SHRINKAGE_LITERATURE_ROADMAP.md`
  - 包含：RF pilot 已证伪（按 time-box 止损）、CFG/位移加权/导航注入的现状与风险点

本文件第 1–6 节主要记录 **dt30 strict 数据闭环** 与 **v1.0（baseline/diffusion/physics）** 的基础结果与排雷清单；如你当前在写 *CFG 版* 报告/论文，请把以上两份文档作为第一入口。

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

> Baseline 口径说明（避免“类不对齐”）：
>
> - **SeqBaseline（deterministic regression）** 是我们的 *Anchor / Prior*（Residual 框架里冻结使用），用于提供 low-frequency mean path；
>   它不是 generative competitor（不具备多模态），因此在主表中应作为 deterministic reference，而不是“必须 beat 的主要对手”。
> - Paper-ready 的主对比应至少包含一个 **同类生成模型 baseline**（例如 CVAE），并使用相同的条件输入 `(obs, o, d, t0)` 与相同的 K-sampling 协议。

- **Deterministic L2 Regression（SeqBaseline, K=1；同时作为 residual prior）**：`src/models/seq/seq_baseline.py`
- **CVAE baseline（多模态，对位 Diffusion/Physics）**：`src/models/seq/seq_cvae.py`
- Data-only Diffusion（ablation：无 nav_field）：`src/models/diffusion/diffusion_model.py`
- Physics Diffusion（nav_patch 条件）：`src/models/physics/physics_condition_diffusion.py`

### 3.2 训练产物与超参（来自 checkpoint config）

权重（本地）：

- Deterministic L2（SeqBaseline）权重：`data/experiments/baseline_b_dt30/last.pt`
- Diffusion：`data/experiments/diff_b_dt30/last.pt`
- Physics：`data/experiments/physics_b_dt30/last.pt`

关键超参（Diffusion/Physics）：

- `obs_len=8`，`pred_len=12`
- `hidden_dim=64`
- `diff_steps=100`
- `patch_size=32`（Physics）
- `epochs=50`，`batch_size=2048`，`lr=1e-3`

Deterministic L2（从权重推断）：

- `hidden_dim=128`
- `data/experiments/baseline_b_dt30/last.pt` 为 legacy 产物（缺少 `config` 字段）；论文版建议重训并记录完整训练配置。

---

## 4. Phase B 当前已有评估结果（preliminary）

### 4.1 评估产物路径

- Quick Eval（320 条 condition；生成模型 K=20，Deterministic L2 为 K=1）：  
  - Deterministic L2：`data/experiments/baseline_b_dt30_eval_quick/metrics.json`（+ `samples.npz`）  
  - Diffusion：`data/experiments/diff_b_dt30_eval_quick/metrics.json`（+ `samples.npz`）  
  - Physics：`data/experiments/physics_b_dt30_eval_quick/metrics.json`（+ `samples.npz`）
- b1 子集（32 条 condition，debug 用的最小闭环，避免跑太久）：  
  - `data/experiments/*_b_dt30_eval_b1/`

### 4.2 微观指标（Quick Eval：320 条 condition，越小越好）

> 说明：Diffusion/Physics 的 `*_std` 是 **每个 condition 内 K 次采样的波动**；`*_best` 为 best-of-K（oracle 上界，用于衡量覆盖潜力）。

| 模型 | K | ADE_mean | ADE_std | ADE_best | FDE_mean | FDE_std | FDE_best | Fréchet_mean | Fréchet_std | Fréchet_best | DTW_mean | DTW_std | DTW_best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic L2 | 1 | 5.467 | 0.000 | 5.467 | 8.855 | 0.000 | 8.855 | 9.234 | 0.000 | 9.234 | 54.496 | 0.000 | 54.496 |
| Diffusion | 20 | 6.741 | 2.651 | 2.836 | 11.636 | 5.025 | 3.950 | 11.838 | 4.956 | 4.433 | 72.064 | 33.670 | 26.194 |
| Physics | 20 | 6.744 | 2.926 | 2.510 | 11.644 | 5.556 | 3.419 | 11.863 | 5.464 | 4.028 | 71.981 | 36.824 | 22.623 |

**现象（必须正视）**：

- 在该 quick 子集上，Diffusion/Physics 的 **mean 口径**微观误差显著劣于 Deterministic L2（ADE/FDE/Fréchet/DTW 均 ↑约 23–32%）。  
- Physics 的 **mean** 与 Diffusion 基本持平，但 **best-of-K** 明显更好（四个指标均比 Diffusion 的 best-of-K 低约 9–14%），说明 nav_field 条件更像是在提升“覆盖潜力/上界”，而不是提升“典型样本质量”。

> 这与 Phase A（Physics 在 mean 口径上优于 Deterministic L2）的趋势不同：Phase B(dt30) 下 Deterministic L2 很强，而 Diffusion/Physics 出现明显的“低位移幅度/偏收缩”问题（见 4.3）。

### 4.3 宏观指标与 GT 对照（Quick Eval：同 320 条 condition）

| 指标 | GT | Deterministic L2 | Diffusion | Physics |
|---|---:|---:|---:|---:|
| MSD_1 | 5.345 | 3.322 | 2.136 | 2.578 |
| MSD_5 | 103.678 | 81.603 | 32.945 | 38.700 |
| MSD_10 | 349.740 | 304.099 | 100.948 | 116.869 |
| Rog | 5.247 | 5.494 | 3.056 | 3.435 |

**宏观解读（preliminary）**：

- Deterministic L2 的 Rog 与 GT 很接近（约 4.7% 相对误差），MSD 也相对更接近（尤其 MSD_10）。  
- Diffusion/Physics 的 MSD 与 Rog 明显偏低，表现为生成轨迹整体“收缩/走不动”，导致微观误差（尤其 FDE）偏大。  
- Physics 相对 Diffusion 的 MSD/Rog 更大（更接近 GT），说明 nav_field 条件确实在“拉回运动幅度”，但目前仍不足以达到可论文结论的水平。

### 4.4 诊断证据：收缩/“走不动”不是猜测（基于保存样本）

> 说明：以下统计来自各模型 quick eval 目录下的 `samples.npz`（默认仅保存 `k=0` 的那条生成样本，`N=200`）。
> 这里的 `path_len` 定义为未来轨迹的累计位移长度 $\sum_t \lVert p_{t}-p_{t-1}\rVert$（单位：grid cell）。

| 模型 | path_len（pred, mean±std） | path_len（GT, mean±std） |
|---|---:|---:|
| Deterministic L2 | 16.466 ± 6.860 | 16.738 ± 10.885 |
| Diffusion | 12.135 ± 7.498 | 16.738 ± 10.885 |
| Physics | 13.190 ± 7.600 | 16.738 ± 10.885 |

结论：Diffusion/Physics 在该子集上确实存在明显“偏短/偏收缩”，Physics 能部分缓解但仍不足以追平 GT 的运动幅度。

---

## 5. 可视化（论文级图件）

Phase B 的“论文级图件”建议与 Phase A 保持同一风格，至少包括：

1. 微观指标对比（ADE/FDE/Fréchet/DTW，mean + best-of-K）
2. MSD 曲线（log-log，横轴 $\tau=k\\cdot 30s$，含 GT 对照与幂律指数）
3. 同一条件下的轨迹叠图（GT + Deterministic L2 + Diffusion + Physics）
4. ADE/FDE 的 CDF（基于保存样本）
5. Rog 的分布（箱线图/小提琴图）
6. 运动幅度诊断（mean step speed / path_len 的箱线图，用于展示“收缩/走不动”）

对应脚本见：`src/visualization/plot_phase_b_report.py`（与 Phase A 脚本保持同风格）。

运行命令（读取 quick 320 的三组评估目录，并输出 PDF+PNG）：

```bash
MPLCONFIGDIR=/tmp/mplconfig \
  python -m src.visualization.plot_phase_b_report \
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

### 6.2 如果 Phase B 仍然“Deterministic L2 > Diffusion/Physics”（需要立刻排雷）

优先排查顺序（KISS）：

1. **采样是否与训练一致**：`diff_steps`、`obs_len/pred_len`、`hidden_dim` 是否对齐 checkpoint（evaluate.py 已自动对齐 hidden_dim）。
2. **容量/批大小是否导致欠拟合**：`hidden_dim=64 + batch_size=2048` 可能偏欠拟合（建议做 `hidden_dim=128`、`batch_size=512/1024` 的对照）。
3. **归一化统计量是否正确**：确认 dt30 的 `data_stats.json` 来自 train split，且训练/评估都读取 dt30 目录下的 stats（目前 sanity check 已 PASS）。
4. **OD 条件是否有效**：检查 cond 编码是否被模型使用（可做 ablation：只用 obs vs obs+OD）。

### 6.3 针对评审建议的快速验证（h=128, batch=512, lr=3e-4, epochs=10）

评审意见见：`docs/archive/phase_b/PHASE_B_REVIEW.md`，核心假设是：
- Phase B 下 Deterministic L2 更容易学到条件均值（strong baseline effect）；
- Diffusion/Physics 可能因容量不足（`hidden_dim=64`）产生欠拟合与“收缩”。

为验证该假设，我们做了最小成本的扩容快速实验（dt30，train split 训练；test quick 评估，320 条 condition）：

- Diffusion 训练：`data/experiments/diff_b_dt30_h128_b512_lr3e-4_e10/last.pt`
- Physics 训练：`data/experiments/physics_b_dt30_h128_b512_lr3e-4_e10/last.pt`
- Diffusion 评估：`data/experiments/diff_b_dt30_h128_b512_lr3e-4_e10_eval_quick/metrics.json`
- Physics 评估：`data/experiments/physics_b_dt30_h128_b512_lr3e-4_e10_eval_quick/metrics.json`

结果（节选，mean 口径；越小越好）：

| 模型 | hidden_dim | epochs | ADE_mean | FDE_mean | Fréchet_mean | DTW_mean | MSD_10 | Rog |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Diffusion（原） | 64 | 50 | 6.741 | 11.636 | 11.838 | 72.064 | 100.948 | 3.056 |
| Physics（原） | 64 | 50 | 6.744 | 11.644 | 11.863 | 71.981 | 116.869 | 3.435 |
| Diffusion（h128,e10） | 128 | 10 | 7.180 | 12.230 | 12.463 | 77.940 | 110.161 | 3.159 |
| Physics（h128,e10） | 128 | 10 | 6.862 | 11.798 | 12.009 | 73.941 | 111.111 | 3.147 |

配套可视化（PDF+PNG）：

```bash
MPLCONFIGDIR=/tmp/mplconfig \
  python -m src.visualization.plot_phase_b_report \
  --baseline_dir data/experiments/baseline_b_dt30_eval_quick \
  --diff_dir data/experiments/diff_b_dt30_h128_b512_lr3e-4_e10_eval_quick \
  --physics_dir data/experiments/physics_b_dt30_h128_b512_lr3e-4_e10_eval_quick \
  --out_dir data/experiments/phase_b_report/figures_h128_e10
```

**解读（preliminary）**：

- 单看 `epochs=10` 的快速验证，扩容并未显著改善 mean 口径，且宏观“收缩”问题仍存在（例如 `path_len`：Diffusion≈12.392，Physics≈12.153，GT≈16.738；来自各自 `samples.npz` 的 `N=200, k=0` 样本）。  
- 该结果并不直接否定“容量不足”假设，更可能说明：**扩容 + 降 lr + 少量 epoch 尚未收敛**，需要跑到与原设置同量级的 epoch（例如 50/100）才能公平验证。

### 6.4 下一轮最小“可证伪”实验（建议直接在 dt30 上做，避免重复已有实验）

目标：用**同一套 dt30 产物**，把“容量不足/未收敛”这个最主要的工程假设先证伪或证实，再决定是否需要更大方法改动。

建议只做最少的 2×3 个训练（Diffusion/Physics × seeds=0/1/2），每个训练后跑 quick + 中等规模评估。  
（走 A 路线时，Deterministic L2 不是主要竞争对手；其多 seed 重训仅作为 *reference*，可选。）

```bash
# 0) 强制使用 dt30 数据（单一真相源）
DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
NAV=data/processed_dt30/nav_field.npz

# 1) 训练：Diffusion / Physics（hidden_dim=128，epochs=100）
#    说明：num_workers 在 Windows/WSL 建议 0；在 Linux 服务器可用 4/8。
for SEED in 0 1 2; do
  # （可选 reference）Deterministic L2 重训（用于检查训练/评估环境是否一致）
  # python -m src.training.train_baseline \
  #   --data_path ${DATA} \
  #   --split train \
  #   --exp_name baseline_b_dt30_h128_b2048_lr1e-3_e50_s${SEED} \
  #   --hidden_dim 128 --batch_size 2048 --lr 1e-3 --epochs 50 \
  #   --num_workers 0 --seed ${SEED}

  python -m src.training.train_diffusion \
    --model_type diffusion \
    --data_path ${DATA} \
    --split train \
    --exp_name diff_b_dt30_h128_b512_lr1e-3_e100_s${SEED} \
    --hidden_dim 128 --batch_size 512 --lr 1e-3 --epochs 100 \
    --num_workers 0 --seed ${SEED}

  python -m src.training.train_diffusion \
    --model_type physics \
    --data_path ${DATA} \
    --nav_file ${NAV} \
    --split train \
    --exp_name physics_b_dt30_h128_b512_lr1e-3_e100_s${SEED} \
    --hidden_dim 128 --batch_size 512 --lr 1e-3 --epochs 100 \
    --num_workers 0 --seed ${SEED}

  # CVAE baseline（多模态，对位 Diffusion/Physics；论文主表建议补齐）
  python -m src.training.train_cvae \
    --data_path ${DATA} \
    --split train \
    --exp_name cvae_b_dt30_h128_z16_b512_lr1e-3_e100_s${SEED} \
    --hidden_dim 128 --latent_dim 16 --beta_kl 0.1 --kl_anneal_epochs 10 \
    --batch_size 512 --lr 1e-3 --epochs 100 \
    --num_workers 0 --seed ${SEED}
done

# 2) 评估：quick（320 条 condition）+ mid（约 6400 条 condition）
for SEED in 0 1 2; do
  # （可选 reference）Deterministic L2 评估（quick/mid）
  # python -m src.training.evaluate \
  #   --exp_name baseline_b_dt30_h128_b2048_lr1e-3_e50_s${SEED}_eval_quick \
  #   --model_type baseline \
  #   --data_path ${DATA} \
  #   --checkpoint data/experiments/baseline_b_dt30_h128_b2048_lr1e-3_e50_s${SEED}/last.pt \
  #   --split test --batch_size 32 --max_batches 10 --num_workers 0 \
  #   --save_samples 200 --seed ${SEED}

  python -m src.training.evaluate \
    --exp_name diff_b_dt30_h128_b512_lr1e-3_e100_s${SEED}_eval_quick \
    --model_type diffusion \
    --data_path ${DATA} \
    --checkpoint data/experiments/diff_b_dt30_h128_b512_lr1e-3_e100_s${SEED}/last.pt \
    --split test --batch_size 32 --max_batches 10 --num_workers 0 \
    --num_samples_per_condition 20 --diff_steps 100 --save_samples 200 --seed ${SEED}

  python -m src.training.evaluate \
    --exp_name physics_b_dt30_h128_b512_lr1e-3_e100_s${SEED}_eval_quick \
    --model_type physics \
    --data_path ${DATA} \
    --checkpoint data/experiments/physics_b_dt30_h128_b512_lr1e-3_e100_s${SEED}/last.pt \
    --nav_file ${NAV} \
    --split test --batch_size 32 --max_batches 10 --num_workers 0 \
    --num_samples_per_condition 20 --diff_steps 100 --save_samples 200 --seed ${SEED}

  python -m src.training.evaluate \
    --exp_name cvae_b_dt30_h128_z16_b512_lr1e-3_e100_s${SEED}_eval_quick \
    --model_type cvae \
    --data_path ${DATA} \
    --checkpoint data/experiments/cvae_b_dt30_h128_z16_b512_lr1e-3_e100_s${SEED}/last.pt \
    --split test --batch_size 32 --max_batches 10 --num_workers 0 \
    --num_samples_per_condition 20 --z_temperature 1.0 --save_samples 200 --seed ${SEED}

  # python -m src.training.evaluate \
  #   --exp_name baseline_b_dt30_h128_b2048_lr1e-3_e50_s${SEED}_eval_mid \
  #   --model_type baseline \
  #   --data_path ${DATA} \
  #   --checkpoint data/experiments/baseline_b_dt30_h128_b2048_lr1e-3_e50_s${SEED}/last.pt \
  #   --split test --batch_size 32 --max_batches 200 --num_workers 0 \
  #   --save_samples 200 --seed ${SEED}

  python -m src.training.evaluate \
    --exp_name diff_b_dt30_h128_b512_lr1e-3_e100_s${SEED}_eval_mid \
    --model_type diffusion \
    --data_path ${DATA} \
    --checkpoint data/experiments/diff_b_dt30_h128_b512_lr1e-3_e100_s${SEED}/last.pt \
    --split test --batch_size 32 --max_batches 200 --num_workers 0 \
    --num_samples_per_condition 20 --diff_steps 100 --save_samples 200 --seed ${SEED}

  python -m src.training.evaluate \
    --exp_name physics_b_dt30_h128_b512_lr1e-3_e100_s${SEED}_eval_mid \
    --model_type physics \
    --data_path ${DATA} \
    --checkpoint data/experiments/physics_b_dt30_h128_b512_lr1e-3_e100_s${SEED}/last.pt \
    --nav_file ${NAV} \
    --split test --batch_size 32 --max_batches 200 --num_workers 0 \
    --num_samples_per_condition 20 --diff_steps 100 --save_samples 200 --seed ${SEED}

  python -m src.training.evaluate \
    --exp_name cvae_b_dt30_h128_z16_b512_lr1e-3_e100_s${SEED}_eval_mid \
    --model_type cvae \
    --data_path ${DATA} \
    --checkpoint data/experiments/cvae_b_dt30_h128_z16_b512_lr1e-3_e100_s${SEED}/last.pt \
    --split test --batch_size 32 --max_batches 200 --num_workers 0 \
    --num_samples_per_condition 20 --z_temperature 1.0 --save_samples 200 --seed ${SEED}
done
```

也可以直接用脚本一键执行（默认只跑 Diffusion/Physics，且会跳过已存在的产物）：`scripts/phase_b_step1_capacity_check.sh`。

---

### 6.5 Step1（三 seed）结果汇总（dt30, h128/b512/lr1e-3/e100）

> 说明：下表的 `mean±std` 是 **跨 seed（n=3）** 的统计（不是 K 采样的 std）。  
> Quick：`num_conditions=320`；Mid：`num_conditions=6400`；`K=20`。

**Quick（320）**

| 模型 | ADE_mean | ADE_best | FDE_mean | FDE_best | Rog | GT_Rog |
|---|---:|---:|---:|---:|---:|---:|
| Diffusion | 7.011 ± 0.600 | 2.684 ± 0.453 | 12.089 ± 1.074 | 3.445 ± 0.618 | 3.317 ± 0.010 | 5.247 |
| Physics | **6.458 ± 0.105** | **2.348 ± 0.126** | **11.303 ± 0.148** | **3.231 ± 0.288** | 3.120 ± 0.110 | 5.247 |

**Mid（6400）**

| 模型 | ADE_mean | ADE_best | FDE_mean | FDE_best | Rog | GT_Rog |
|---|---:|---:|---:|---:|---:|---:|
| Diffusion | 8.579 ± 0.458 | 3.264 ± 0.471 | 14.726 ± 0.735 | 4.539 ± 0.547 | 3.834 ± 0.050 | 6.533 |
| Physics | **8.129 ± 0.124** | **2.984 ± 0.041** | **14.130 ± 0.222** | **4.369 ± 0.122** | 3.724 ± 0.056 | 6.533 |

**解读（可写进报告/论文的“排雷结论”）**

- Physics 相对 Diffusion：在 **mean 口径**与 **best-of-K** 上均稳定更优（quick/mid 一致），且 best-of-K 的跨 seed 方差明显更小（更稳定的“覆盖上界”）。  
- 但宏观 Rog 明显低于 GT（约 0.57–0.63×），说明 **生成分布仍存在收缩/走不动**；Physics 的 Rog 还略低于 Diffusion，提示 nav_field 可能在“提升上界”的同时把典型样本推向更保守的运动幅度，需要后续用分布指标（Energy Score/CRPS）+ 选样策略/正则进一步解决。

---

### 6.6 收缩问题的修正：Temperature ≠ Scale（必须解耦）

**结论（先写死，避免误用）**：

- **Temperature（采样噪声/随机性）**主要影响 *抖动/多样性*，不应被用来“撑大”轨迹位移幅度；提高 temperature 往往优先增加 high-frequency jitter（曲折、毛刺），会显著改变 `path_len`，但对 `Rog/MSD` 的提升不可控，且常导致 ADE/FDE 变差。
- **Scale（`vel_scale`）**是对症下药：对 future `vel`（step displacement）做系统性缩放，直接控制物理幅度。它不会引入额外抖动，只改变整体运动尺度：
  - `path_len` ∝ `vel_scale`
  - `Rog` ∝ `vel_scale`
  - `MSD` ∝ `vel_scale^2`

因此，解决收缩必须优先用 `vel_scale`（或训练时的幅度正则），而不是 temperature。

#### 6.6.1 可复现的 `vel_scale` 校准流程（推荐：val→test）

1) **在 val split 上评估（vel_scale=1.0）**：

```bash
DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
NAV=data/processed_dt30/nav_field.npz

# Diffusion
python -m src.training.evaluate \
  --exp_name diff_b_dt30_step1_val_eval \
  --model_type diffusion \
  --data_path ${DATA} \
  --checkpoint data/experiments/diff_b_dt30_h128_b512_lr1e-3_e100_s0/last.pt \
  --split val --batch_size 32 --max_batches 200 --num_workers 0 \
  --num_samples_per_condition 20 --diff_steps 100 --save_samples 0 --seed 0

# Physics
python -m src.training.evaluate \
  --exp_name physics_b_dt30_step1_val_eval \
  --model_type physics \
  --data_path ${DATA} \
  --checkpoint data/experiments/physics_b_dt30_h128_b512_lr1e-3_e100_s0/last.pt \
  --nav_file ${NAV} \
  --split val --batch_size 32 --max_batches 200 --num_workers 0 \
  --num_samples_per_condition 20 --diff_steps 100 --save_samples 0 --seed 0
```

2) **用校准工具计算推荐 `vel_scale`（优先 speed/path_len）**：

```bash
python -m src.utils.calibrate_vel_scale \
  data/experiments/diff_b_dt30_step1_val_eval/metrics.json \
  --prefer speed
```

3) **固定该 `vel_scale`，在 test split 复现评估**（这一步才写进论文主表，避免泄漏）：

```bash
python -m src.training.evaluate \
  --exp_name diff_b_dt30_step1_test_eval_scaled \
  --model_type diffusion \
  --data_path ${DATA} \
  --checkpoint data/experiments/diff_b_dt30_h128_b512_lr1e-3_e100_s0/last.pt \
  --split test --batch_size 32 --max_batches 200 --num_workers 0 \
  --num_samples_per_condition 20 --diff_steps 100 --save_samples 200 --seed 0 \
  --vel_scale <FILL_FROM_CALIBRATION>
```

> 备注：若 `vel_scale` 能明显拉近 `gt_speed_mean/gt_path_len_mean` 但 `Rog/MSD` 仍偏小，则说明问题不止是尺度，还包含“时间相关性/方向持久性”不足；此时应进入训练级修复（训练期 Macro Loss）。  
> 重要：Macro Loss 必须做 diffusion timestep 门控（例如 `t < 50`），否则在大噪声步上约束 `x0_pred` 会触发高频抖动爆炸（path_len 变大但净位移更差）。  
> 代码入口（推荐）：`python -m src.training.train_diffusion --lambda_rog <W> --macro_metric epe --macro_t_threshold 50 --rog_warmup_epochs 5`（默认 Macro 关闭；Diffusion/Physics 都支持）。

#### 6.6.2 校准后的效果（test-mid, 6400 conditions，示例：seed2）

**结论（事实）**：`vel_scale` 能把 *速度/路径长度/活动半径* 拉到接近 GT，但 **微观误差（ADE/FDE/Fréchet/DTW）会显著变差**；这说明 Phase B 的问题不仅是“幅度收缩”，还包含“方向/时间相关性不足（更像随机游走）”，尺度放大后方向误差被同步放大。

示例产物（test-mid，`num_conditions=6400, K=20`）：

- Diffusion（scale 后）：`data/experiments/diff_dt30_s2_eval_test_mid_velscale/metrics.json`
- Physics（scale 后）：`data/experiments/phys_dt30_s2_eval_test_mid_velscale/metrics.json`
- Diffusion（scale 前，对照）：`data/experiments/diff_b_dt30_h128_b512_lr1e-3_e100_s2_eval_mid/metrics.json`
- Physics（scale 前，对照）：`data/experiments/physics_b_dt30_h128_b512_lr1e-3_e100_s2_eval_mid/metrics.json`

**宏观幅度（越接近 GT 越好）**：

| 模型 | vel_scale | pred_speed_mean / gt_speed_mean | Rog | GT_Rog | MSD_10 | GT_MSD_10 |
|---|---:|---:|---:|---:|---:|---:|
| Diffusion（scale 前） | 1.0 | — | 3.863 | 6.533 | 159.33 | 505.67 |
| Diffusion（scale 后） | 1.6395 | 2.069 / 1.980 ≈ 1.045 | 6.328 | 6.533 | 425.21 | 505.67 |
| Physics（scale 前） | 1.0 | — | 3.797 | 6.533 | 156.47 | 505.67 |
| Physics（scale 后） | 1.6804 | 2.062 / 1.980 ≈ 1.041 | 6.383 | 6.533 | 439.42 | 505.67 |

**微观指标（越小越好；scale 会变差）**：

| 模型 | ADE_mean | FDE_mean | DTW_mean |
|---|---:|---:|---:|
| Diffusion（scale 前） | 8.329 | 14.262 | 88.974 |
| Diffusion（scale 后） | 9.561 | 16.041 | 101.468 |
| Physics（scale 前） | 8.107 | 14.110 | 85.834 |
| Physics（scale 后） | 9.370 | 16.043 | 98.512 |

> 关键现象：即使 `pred_speed_mean` 已接近 `gt_speed_mean`，`MSD_10` 仍低于 GT（约 0.84–0.87×）。这说明轨迹虽然“走得动”，但仍存在明显的方向抵消/转向过频（缺少方向持久性），属于训练级问题，不能靠单一 `vel_scale` 完全解决。

---

## 7. Phase B v1.1：Residual Diffusion（结构性修复 shrinkage）

> 背景：v1.0/v1.0+macro/vel_scale 的证据链表明，“收缩”不只是校准问题，而是生成分布学习失败 + 物理先验保守性叠加。  
> 核心改动（KISS）：用确定性 baseline 作为 **prior** 固定低频结构与尺度，让 diffusion/physics 只学习 residual（不再从零学“物理 + 随机性”）。

### 7.1 方案定义（单一入口）

详见：`docs/archive/phase_b/RESIDUAL_DIFFUSION.md`

### 7.2 已完成的 fast 证据（test, K=10，确认方向）

> 说明：以下为 fast eval（用于确认“宏观是否恢复 + micro 是否有增益”），不是最终 paper table。

| 模型 | 指标文件 | `pred_speed_mean/gt_speed_mean` | `Rog/GT_Rog` | `MSD_10/GT_MSD_10` | `ADE_best` | `FDE_best` |
|---|---|---:|---:|---:|---:|---:|
| Residual Diffusion | `data/experiments/diff_residual_test_fast/metrics.json` | 1.016 | 1.004 | 0.935 | 3.656 | 4.975 |
| Residual Physics | `data/experiments/phys_residual_test_fast/metrics.json` | 0.951 | 0.945 | 0.864 | 2.709 | 3.569 |

**解读（事实 + 最小推论）**：
- Residual（data-only）在 fast eval 中 **几乎完全恢复宏观尺度**（speed/Rog 接近 1），说明“prior+residual decomposition”对 shrinkage 是结构性修复。
- Residual Physics micro 更强（best-of-K 更好），但宏观仍略保守（MSD10≈0.86×GT），与 nav_field 的“local mean-flow tether”一致；这提示 physics-conditioned residual 仍需隔离变量继续排查（但方向已经显著优于 v1.0）。

### 7.3 下一步（时间成本可控的验证顺序）

1) 先按 `docs/archive/phase_b/RESIDUAL_DIFFUSION.md` 的建议做两阶段评估（`K=1+B=50` → `K=10+B=200`）。  
2) 只在“方向正确”的前提下再跑全量 test（否则纯属烧时间）。  
3) 若 physics residual 仍偏保守，优先做 **conditioning 的低成本 ablation**（例如 `nav_emb_scale`），但避免无意义的大 sweep（已观察到收益快速饱和）。

# Residual Diffusion（v1.1）：用 Deterministic Baseline 做 Prior

本说明用于解决 v1.0 中观察到的 **Mean-Reversion Shrinkage**：纯 diffusion / physics diffusion 的典型速度偏小（macro: MSD/Rog 偏低）。

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

## 7) 评估成本控制（强烈建议）

为了避免“跑一晚才知道没效果”，建议统一使用两阶段评估：

1) **粗筛**（确认方向）：`K=1, max_batches=50`
2) **精验**（确认趋势）：`K=10, max_batches=200`
3) **最终**（paper）：`K=20, full test`（或至少 `max_batches=2000`）

并行建议：
- 同一台多 GPU：每张卡跑一个 eval（不同 `exp_name`），用 `python -u ... |& tee` 保留实时进度。
- HDF5 并行不稳时：设置 `HDF5_USE_FILE_LOCKING=FALSE`，必要时 `--num_workers 0`。

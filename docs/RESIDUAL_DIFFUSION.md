# Residual Diffusion（v1.1）：用 Deterministic Baseline 做 Prior

本说明用于解决 v1.0 中观察到的 **Mean-Reversion Shrinkage**：纯 diffusion / physics diffusion 的典型速度偏小（macro: MSD/Rog 偏低）。

核心思路（KISS）：

- 冻结一个确定性模型（`SeqBaseline`）作为 **prior**：$\hat{v}_{prior} = f_{base}(\text{obs}, \text{cond})$
- diffusion/physics 不再直接建模 $v$，而是建模 **残差**：
  $$v = v_{prior} + v_{residual}$$

代码支持：

- `src/training/train_diffusion.py`：新增 `--prior_checkpoint`，训练时自动用 `vel_full - vel_prior` 作为 target（diff loss 训练 residual；macro loss 若启用则在 full trajectory 上计算）
- `src/training/evaluate.py`：新增 `--prior_checkpoint`，评估时自动将采样到的 residual 加回 prior

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


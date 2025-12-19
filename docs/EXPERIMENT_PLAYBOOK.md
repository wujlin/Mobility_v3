Implementation Plan, Task List and Thought in Chinese：本文件是一份“少踩坑”的实验运行手册（面向多机器/多 GPU/时间紧的情况），目标是 **更快证伪、更少返工、更可复现**。

# Experiment Playbook（少踩坑版）

> 适用：Phase A/B 训练 + 评估 + 可视化。  
> 原则：先做 *fast check*，再做 *confirm*，最后才做 *full test*。

---

## 0) Preflight（开跑前 60 秒）

1) **确认数据与产物路径存在**（缺一个就会直接退出）：

```bash
DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
NAV=data/processed_dt30/nav_field.npz
PRIOR=data/experiments/baseline_b_dt30/last.pt
test -f "$DATA" && test -f "$NAV" && test -f "$PRIOR"
```

2) **确认 checkpoint 在当前机器上存在**（跨机器最常见坑）：

```bash
CKPT=data/experiments/<exp_name>/last.pt
test -f "$CKPT"
```

3) **确认 split 文件存在且一致**：

```bash
ls -1 data/processed_dt30/splits/{train_ids,val_ids,test_ids}.npy
```

---

## 1) tmux/日志（不要把进度藏起来）

推荐方式：实时输出 + 同步落盘。

```bash
mkdir -p logs
PYTHONUNBUFFERED=1 python -u -m src.training.train_diffusion ... |& tee logs/train_xxx.log
```

查看进度：

```bash
tail -f logs/train_xxx.log
```

并行跑时（两张 GPU）：

```bash
CUDA_VISIBLE_DEVICES=0 ... |& tee logs/job0.log &
CUDA_VISIBLE_DEVICES=1 ... |& tee logs/job1.log &
wait
```

---

## 2) HDF5 并行（卡锁/卡顿的最小修复）

经验：并行 eval + dataloader 多进程时最容易出问题。

优先设置：

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

若仍不稳定，直接降级（牺牲一点速度换稳定）：

- 评估：`--num_workers 0`
- 训练：`--num_workers 4/8` 逐步加，不要一上来 16

---

## 3) 评估成本控制（强烈建议统一两阶段）

### 3.1 Fast Check（确认方向，10–20 分钟级）

- `K=1`
- `max_batches=50`（或更小）
- `save_samples=0`

示例：

```bash
python -m src.training.evaluate \
  --exp_name <exp>_val_k1_fast \
  --model_type physics \
  --data_path ${DATA} --nav_file ${NAV} \
  --checkpoint ${CKPT} \
  --split val \
  --batch_size 512 --num_workers 0 --max_batches 50 \
  --num_samples_per_condition 1 --diff_steps 100 --save_samples 0 --seed 0
```

### 3.2 Confirm（确认趋势，1 小时级）

- `K=10`
- `max_batches=200`

### 3.3 Full Test（paper）

- `K=20`
- `full test`（或至少 `max_batches=2000`）

---

## 4) Residual 模式（v1.1）专用检查

最常见坑：评估忘记加 prior，导致你评估的是 residual 本身（必然非常小）。

```bash
python -m src.training.evaluate ... \
  --prior_checkpoint ${PRIOR}
```

快速 sanity 指标（优先看 ratio 而不是绝对值）：

- `pred_speed_mean / gt_speed_mean`
- `Rog / GT_Rog`
- `MSD_10 / GT_MSD_10`

---

## 5) 多机器同步（校园网常用拓扑）

典型拓扑（你们当前的真实约束）：
- 本地电脑 ⇄ 工作站A（可直连）
- 服务器B → 工作站A（单向可直连）

因此建议统一走“B → A → 本地”的 rsync 流水线。

### 5.1 从服务器B推到工作站A

（在服务器B执行）

```bash
rsync -avP data/experiments/<exp_dir>/ \
  jinlin@10.13.12.164:/home/jinlin/projects/Mobility_v3/data/experiments/<exp_dir>/
```

### 5.2 从工作站A拉回本地

（在本地执行；若需 socks5 代理，用 ssh config 的 `ProxyCommand`）

```bash
rsync -avP wsA:/home/jinlin/projects/Mobility_v3/data/experiments/<exp_dir>/ \
  data/experiments/<exp_dir>/
```

---

## 6) 命名与记录（可复现最小集）

建议每个实验目录至少包含：
- `last.pt`
- `metrics.json`（评估产物）
- `logs/*.log`（训练日志，至少能看到 epoch/batch/关键超参）

并在 `exp_name` 中编码关键变量：
- 数据：`dt30`
- 架构：`residual_priorB` / `physics`
- 超参：`h128_b2048_lr1e-3_e100`
- seed：`s0/s1/s2`

---

## 7) 什么时候该停？（避免无意义烧时间）

出现以下任何一种情况，优先停止 full run，回到 fast check：
- `pred_speed_mean/gt_speed_mean < 0.8` 且 10 分钟内无改善迹象（大概率收缩仍在）
- ablation 的收益“单调但饱和”（例如 `nav_emb_scale` 从 1.0→1.25 几乎不变）
- 微观指标全面恶化且宏观无显著改善（说明改动方向不对）


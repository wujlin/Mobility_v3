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

## 2.1) 计算资源怎么“吃满”（少走弯路版）

> 目标：同样的 wall time，拿到更可靠的证伪/证据。  
> 原则：**先把数据喂饱 GPU**，再谈 batch_size/学习率这种二阶问题。

### (A) 先固定 CPU 线程，避免“线程打架”

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
```

### (B) `num_workers` 优先级 > `batch_size`

我们在 dt30 的 *physics* 训练上做过小基准（`--max_batches 200` 固定步数）：

- `batch_size=1024, num_workers=0`：~272s/epoch（明显“饿 GPU”）
- `batch_size=1024, num_workers=8`：~45s/epoch（明显加速）

**建议默认值**：
- 训练：`--num_workers 8`（不稳定再降到 4）
- 评估：优先 `--num_workers 0`（避免多进程 + HDF5 的偶发锁/卡顿）

### (C) 说明：启用 `--max_batches` 时，batch_size 过大反而更慢

如果你为了快速验证启用了 `--max_batches`（例如每个 epoch 只跑 200 个 batch），那么：

- **batch_size 越大，每个 batch 计算量越大** → epoch wall time 变长；
- 因为你本来就不跑完整 epoch，所以“大 batch 缩短 epoch step 数”的优势并不存在。

因此在 *fast training*（带 `--max_batches`）时，推荐：
- `batch_size=1024` 或 `2048`（先用 1024）
- 如果 GPU 仍然空闲，再逐步加到 2048（不要一步上 8192/16384）

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

### 3.4 Baseline 训练也支持快跑（prior 调参/排雷用）

为了避免 deterministic prior（baseline）的验证也要每次跑满全量，我们在 `src/training/train_baseline.py` 加入了 `--max_batches`（默认不启用，不影响原实验）。

示例（每个 epoch 只跑 200 个 batch，用于快速验证 displacement-aware weighting 是否方向正确）：

```bash
python -m src.training.train_baseline \
  --exp_name baseline_dt30_dispw_smoke \
  --data_path ${DATA} --split train \
  --batch_size 1024 --epochs 3 --max_batches 200 \
  --disp_weight clip --disp_clip_min 0.5 --disp_clip_max 5.0 \
  --num_workers 8 --seed 0
```

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

### 5.3 本地通过 EasyConnect（SOCKS5 1080）访问工作站A（最常见坑）

如果你的本地环境被 Docker/EasyConnect 隔离，需要通过 SOCKS5 代理（如 `127.0.0.1:1080`）才能访问校园网内网主机：

1) **把配置写进 `~/.ssh/config`（不要在 shell 里直接敲 `Host ...`）**

```sshconfig
Host wsA
    HostName 10.13.12.164
    User jinlin
    ProxyCommand nc -X 5 -x 127.0.0.1:1080 %h %p
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

2) **快速连通性测试（优先做这个，30 秒排雷）**

```bash
# 代理是否工作：能否通过 socks 连到 wsA 的 22 端口
nc -X 5 -x 127.0.0.1:1080 10.13.12.164 22

# ssh 是否能走 wsA 连接
ssh -v wsA "hostname"
```

出现 `SOCKSv5 error: TTL expired` / `Connection closed` 往往意味着：
- EasyConnect/代理没有起来，或 1080 端口在当前容器/WSL 里不可达；
- 代理起来了，但代理侧路由不到 `10.13.12.164:22`（需要重连/重启 EasyConnect，或检查防火墙策略）。

3) **不依赖 ssh config 的 rsync（更稳，适合临时环境）**

```bash
rsync -avP -e "ssh -o ProxyCommand='nc -X 5 -x 127.0.0.1:1080 %h %p'" \
  jinlin@10.13.12.164:/home/jinlin/projects/Mobility_v3/data/experiments/<exp_dir>/ \
  data/experiments/<exp_dir>/
```

> 重要：`127.0.0.1:1080` 是“运行 rsync/ssh 的那台机器”的本地环回地址。  
> 如果你在服务器B执行 rsync，就不能指望它使用你本地电脑的 1080 代理（除非服务器B上也有同样的代理服务）。

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

补充（physics residual 常见瓶颈）：
- 如果 macro 指标卡在 `RoG/MSD10≈0.93–0.95` 且继续训练只会让 micro 更好、macro 更差（safe-play），不要继续拉长 epoch；优先转向 conditioning 注入的结构性 ablation（例如 `--nav_gate obscond`，learnable gating 用于减弱 mean-field tether）。

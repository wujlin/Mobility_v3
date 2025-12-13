# 任务定义与实验协议（两阶段：fast → paper）

> **适用范围**：本仓库所有训练/评估/数据产物必须遵循本规范  
> **更新**：2025-12-13  
> **核心目标**：消除“文档—代码—数据产物”不一致，保证 **KnownDestination + 无泄漏 + 可复现**

---

## 0. 两阶段路线（避免“先做错再重做”）

**Phase A：fast validation（当前实现，1–2 周）**
- **目的**：验证 pipeline 正确性 + Physics 是否有提升趋势 + 评估闭环是否完整
- **任务定义**：KnownDestination；推理时 `d` 是合法输入，不属于泄漏
- **建模对象**：窗口级未来段生成（带观测历史），而非“一次性生成整段 trip”
- **`vel` 语义**：`step displacement`（步位移），`vel = pos[t] - pos[t-1]`，单位 `grid_cell/step`
- **dt 处理**：不强制重采样；仅记录 dt 分布用于诊断（不能做严格物理时间结论）
- **无泄漏**：训练/评估按 split 过滤；`data_stats.json/nav_field.npz` 仅用 train split 估计并记录 `source/metadata`

**Phase B：paper strict（论文版，2–4 周，必须重训）**
- **目的**：方法论严谨 + 实验可复现 + 结论站得住
- **dt 处理**：必须重采样到固定 `dt_fixed=30s`
- **其余保持一致**：KnownDestination + step displacement（每一步对应 30s）+ train-only 产物合同 + split-aware 训练/评估

---

## 1. 任务定义：Known vs Unknown Destination

| 设定 | 推理时条件 | 适用场景 | 泄漏风险 |
|---|---|---|---|
| **KnownDestination** | 已知 `(o, d, t0)` | 路径生成 / 导航 / 条件生成 | 低（`d` 是输入） |
| **UnknownDestination** | 只知 `(o, t0)` | 轨迹预测 / 异常检测 | 高（若训练使用了 `d`） |

### 1.1 Phase A/B 共用：KnownDestination（带观测历史）

本仓库当前训练/推理的最小闭环任务是 **窗口级未来段生成**：

给定：
- `obs`：历史观测窗口（长度 `H`），每步包含 `[pos, vel]`
- `o`：该 trip 的起点（trip-level origin）
- `d`：该 trip 的终点（trip-level destination）
- `t0`：该 trip 的出发时间（trip-level start time）
- `env`：可选环境特征（v1 主要是 `nav_patch`）

学习：

$$P(\\mathrm{vel}_{t+1:t+F} \\mid \\mathrm{obs}_{t-H+1:t}, o, d, t_0, env)$$

其中：
- `pos` 采用栅格坐标 `[y, x]`
- `vel` 为步位移（见第 2 节）
- 输出是未来 `F` 步 `vel` 序列；位置序列通过积分得到：
  - `pos_pred[k] = pos_last + sum_{i=1..k} vel_pred[i]`

> **注意**：这不是“一次性生成整段 trip”。若要做整段路径生成（one-shot 或 autoregressive），属于后续版本扩展。

---

## 2. 时间与 `dt/vel` 语义

### 2.1 时间戳与时间特征（t0 encoding）

- HDF5 中 `timestamps` 为 Unix 秒（`int64`）
- 时间特征以 **Asia/Shanghai（UTC+8）** 计算（hour/weekday）
- v1 strict 的时间条件向量目前保持 **cond_dim=6**，其中时间部分为 2 维：
  - `hour_norm = hour / 23`
  - `weekday_norm = weekday / 6`（Monday=0）

> 若需要更合理的周期编码（`sin/cos` + `is_weekend`），会改变 `cond_dim`，需要同步修改模型与重训（建议作为 v2 / v1.1）。

### 2.2 `dt` 的边界与论文版要求（关键）

- 原始出租车 GPS 采样间隔 **不固定**（常见 10–60s，且存在更大 gap）
- **Phase A（fast）**：不强制重采样；将每个点视为一个离散 step，并记录 `dt` 分布用于诊断（`data_stats.json.time_stats.dt_stats_sample`）
  - 可以做 step-based 的 sanity 与趋势验证
  - 不建议把 MSD 的横轴解释为真实 $\Delta t$，避免审稿质疑
- **Phase B（paper）**：必须重采样到固定 `dt_fixed`（建议 30s）
  - 这样 MSD 的 $\Delta t = k \\, dt_{fixed}$ 才有明确物理意义
  - nav_field 的 speed（位移模长）也才可跨数据集/时间段对比

### 2.3 `vel` 的唯一语义（决策 B）

```python
# vel: 步位移（step displacement）
vel[t] = pos[t] - pos[t-1]         # 单位: grid_cell/step

# 需要物理速度时（可选）
physical_velocity[t] = vel[t] / dt # 单位: grid_cell/second
```

---

## 3. 数据划分与无泄漏原则（必须执行）

### 3.1 split 文件（唯一真相源）

```
data/processed/splits/
  train_ids.npy
  val_ids.npy
  test_ids.npy
```

### 3.2 严格规则

必须仅用 **train split** 估计：
- `data_stats.json`（normalizer 统计量）
- `nav_field.npz`（direction/speed/count）

训练与评估必须按 split 过滤轨迹：
- 训练默认 `--split train`
- 评估默认 `--split test`

### 3.3 数据产物合同（实际落地字段）

`data/processed/data_stats.json`（示例字段）：

```json
{
  "created_at": "...",
  "source": {
    "split": "train",
    "trajectory_ids_file": "splits/train_ids.npy",
    "trajectory_ids_sha256": "...",
    "trajectories_h5_file": "processed/trajectories/shenzhen_trajectories.h5",
    "trajectories_h5_sha256": "...",
    "date_range": ["...", "..."]
  },
  "grid_config": {"H": 400, "W": 800, "...": "..."},
  "normalization": {"pos_min": [...], "pos_max": [...], "vel_mean": [...], "vel_std": [...], "nav_scale": 1.0, "nav_max_speed": 20.0},
  "time_stats": {"dt_stats_sample": {...}}
}
```

`data/processed/nav_field.npz`：
- `direction`: `(2, H, W)`（内部统一为 `[dir_y, dir_x]`）
- `speed`: `(H, W)`（平均步位移模长）
- `count`: `(H, W)`（样本数）
- `metadata`: dict（npz object，需要 `allow_pickle=True` 加载）

---

## 4. 评估协议（v1）

### 4.1 采样设置（生成模型）

- 生成模型：每个条件采样 `K=20`
- 确定性 baseline：`K=1`（或重复 K 次但 std=0，本质等价）

### 4.2 微观指标（窗口级）

- ADE：平均位移误差
- FDE：终点位移误差
- Fréchet：离散 Fréchet 距离（轨迹形状距离）
- DTW：Dynamic Time Warping 距离（允许时间对齐的形状距离）
- 报告口径（生成模型）：`mean / std / best-of-K`
  - `best-of-K`：对每条样本取 K 条生成里误差最小的那条，再在 batch 上取平均

> v1 的窗口预测长度通常不足以到达 trip 终点，因此 **Arrival Rate（到达率）不作为 v1 默认指标**；若要做需要定义“到达”与 rollout 策略。

---

## 5. nav_field 规范与常见误解

### 5.1 语义

- `nav_field` 是 **目的地无关** 的经验方向/速度先验（更像“道路局部方向”），不是“指向目的地的势场/最短路场”
- 因为道路常有双向流，`mean cos` 可能偏低；更建议同时报告 `mean|cos|` 做一致性诊断

### 5.2 估计与对齐检查

- 估计：仅用 train split，对每个格子统计步位移向量均值并单位化
- 对齐检查：使用 `src/utils/sanity_check.py`，关注：
  - `mean_cos` 与 `mean|cos|`
  - `count>=min_count` 的过滤结果

---

## 6. 推荐命令（Phase A：fast validation / strict no-leak）

生成严格数据产物（train-only）：

```bash
python -m src.data.build_strict_products --processed_dir data/processed
```

严格 sanity check：

```bash
python -m src.utils.sanity_check --data_path data/processed --strict
```

训练（按 split）：

```bash
python -m src.training.train_diffusion \
  --data_path data/processed/trajectories/shenzhen_trajectories.h5 \
  --model_type physics \
  --nav_file data/processed/nav_field.npz \
  --split train \
  --exp_name physics_v1_strict
```

评估（按 split，生成模型默认 K=20）：

```bash
python -m src.training.evaluate \
  --exp_name physics_v1_strict_eval \
  --model_type physics \
  --data_path data/processed/trajectories/shenzhen_trajectories.h5 \
  --checkpoint data/experiments/physics_v1_strict/last.pt \
  --nav_file data/processed/nav_field.npz \
  --split test \
  --num_samples_per_condition 20
```

---

## 7. Phase B：论文版严格协议（dt=30s 重采样）

> **目标会议**：顶会/子刊（NeurIPS, ICML, KDD, AAAI 等）

### 7.1 为什么需要 dt 重采样？

| 问题 | 不重采样的风险 | 审稿人可能质疑 |
|-----|--------------|--------------|
| MSD 标度律 | $\langle \Delta r^2 \rangle \sim \Delta t^\alpha$ 的 $\Delta t$ 含义不明 | "MSD 指数的物理意义？" |
| 速度场语义 | nav_field 的 "速度" 是位移，不同采样间隔不可比 | "nav_field 如何保证一致性？" |
| 宏观正则 | 基于 MSD 的 loss 没有物理基础 | "物理约束真的是物理吗？" |
| 可复现性 | 不同数据源采样间隔不同 | "换数据集还能用吗？" |

### 7.2 严格版本要求（必须写死并进入产物合同）

- **dt_fixed**：30 秒（或其他固定值，但必须写入数据产物合同）
- **重采样方法**：线性插值（在 grid 空间分别对 y/x 插值）
- **重复/乱序时间戳**：必须定义可复现处理（例如：同一秒多点取均值；非单调直接丢弃该 trip）
- **gap 处理**：必须制定可复现规则（例如 `max_gap=300s`，超过则丢弃该 trip 或 split 成多条）
- **vel 语义保持不变（决策 B）**：仍用 `step displacement`
  - 只是每一步对应 `dt_fixed`，因此需要物理速度时：`physical_velocity = vel / dt_fixed`
- **数据产物建议独立目录**（避免覆盖 Phase A）：
  - `data/processed_dt30/trajectories/shenzhen_trajectories.h5`
  - `data/processed_dt30/splits/*.npy`
  - `data/processed_dt30/data_stats.json`（train-only）
  - `data/processed_dt30/nav_field.npz`（train-only）

### 7.3 工程落地（当前缺口与可复现闭环）

1) **生成 dt-fixed 数据集（已实现）**：输入 Phase A 的 HDF5 + splits，输出新的 HDF5 + splits，并写入可复现合同（`resample_meta.json` + old/new id 映射）：

```bash
python -m src.data.build_dt_fixed_dataset \
  --input_processed_dir data/processed \
  --output_processed_dir data/processed_dt30 \
  --dt_fixed 30 \
  --max_gap 300 \
  --min_length 10
```

> 当前实现的 gap 策略：**drop 整条轨迹**（超过 `max_gap` 直接丢弃），以保持 trip-level OD 语义一致性（KnownDestination）。

2) **复用现有严格产物生成器（train-only，无泄漏）**：

```bash
python -m src.data.build_strict_products --processed_dir data/processed_dt30 --backup
python -m src.utils.sanity_check --data_path data/processed_dt30 --strict --expected_dt 30 --dt_require_constant
```

- 论文版训练/评估统一指向 `data/processed_dt30/...`，并固定随机种子与配置日志

### 7.4 论文实验设计

**三层模型对比**（核心贡献）：

| 模型 | 描述 | 物理约束 |
|-----|------|---------|
| Seq Baseline | RNN/Transformer 序列预测 | 无 |
| Data-only Diff | 纯数据扩散生成 | 无 |
| Physics Diff | 物理约束扩散 | nav_field + (可选) macro reg |

**三层评估指标**：

| 层次 | 指标 | 说明 |
|-----|------|-----|
| 微观 | ADE, FDE, Fréchet, DTW（mean/std/best-of-K） | 单条轨迹误差 |
| 中观 | (v2) 路径分布, OD 匹配 | 需要 road-level |
| 宏观 | MSD 曲线, Rog 分布 | 物理是否在帮忙 |

**消融实验**：

| 实验 | 目的 |
|-----|------|
| Physics vs Data-only | nav_field 的贡献 |
| w/ vs w/o destination | KnownDest 的影响 |
| dt=30s vs step-based | 重采样的影响 |
| K=5,10,20 | 采样数敏感性 |


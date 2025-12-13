# 任务定义与实验协议（v1 strict）

> **适用范围**：本仓库所有训练/评估/数据产物必须遵循本规范  
> **更新**：2025-12-13  
> **核心目标**：消除“文档—代码—数据产物”不一致，保证 **KnownDestination + 无泄漏 + 可复现**

---

## 0. v1 strict 总结（先把口径锁死）

- **任务定义**：KnownDestination；推理时 `d` 是合法输入，不属于泄漏
- **建模对象**：窗口级未来段生成（带观测历史），而非“一次性生成整段 trip”
- **`vel` 语义**：`step displacement`（步位移），`vel = pos[t] - pos[t-1]`，单位 `grid_cell/step`
- **数据划分**：`data/processed/splits/{train,val,test}_ids.npy`；训练/评估必须按 split 过滤轨迹
- **严格数据产物**：`data_stats.json` 与 `nav_field.npz` **仅用 train split** 估计，并写入 `source/metadata`

---

## 1. 任务定义：Known vs Unknown Destination

| 设定 | 推理时条件 | 适用场景 | 泄漏风险 |
|---|---|---|---|
| **KnownDestination** | 已知 `(o, d, t0)` | 路径生成 / 导航 / 条件生成 | 低（`d` 是输入） |
| **UnknownDestination** | 只知 `(o, t0)` | 轨迹预测 / 异常检测 | 高（若训练使用了 `d`） |

### 1.1 v1 strict 采用：KnownDestination（带观测历史）

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

### 2.2 `dt` 现状与处理边界

- 原始出租车 GPS 采样间隔 **不固定**（常见 10–60s，且存在更大 gap）
- **v1 strict 不强制重采样**：将每个点视为一个离散 step，并记录 `dt` 分布用于诊断（`data_stats.json.time_stats.dt_stats_sample`）
- 若后续要做严格物理速度或 MSD 标度律的“物理时间”解释，必须先将轨迹重采样到固定 `dt_fixed`（推荐单独产出新的 trajectories 文件并重训）

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
  "normalization": {"pos_min": [...], "pos_max": [...], "vel_mean": [...], "vel_std": [...], "nav_scale": 1.0},
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

## 6. 推荐命令（严格版本）

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

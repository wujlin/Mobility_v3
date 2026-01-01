# 任务定义与实验协议（当前主线：WorldTrace × Detroit｜兼容 legacy dt30）

> **适用范围**：本仓库所有训练/评估/数据产物必须遵循本规范。  
> **更新**：2026-01-01  
> **核心目标**：消除“文档—代码—数据产物”不一致，保证 **KnownDestination + 无泄漏 + 可复现**。  
> **重要说明**：本仓库目前同时保留两条口径：
> - **Phase D（当前主线）**：WorldTrace × Detroit（1Hz、WGS84、matched 坐标、OSM 软先验）
> - **Phase C / Legacy**：深圳出租车 dt30（用于复现历史结论，不再作为主线）

---

## 0. 任务口径总览（避免“口径漂移导致无法归因”）

### Phase D（当前主线）：WorldTrace × Detroit（端到端 + 软先验）

- **主目标**：学习“trip-level 决策 + 执行”的统一生成框架，但不把外部地图当真值。  
- **任务定义**：KnownDestination（推理时 `d` 是合法输入，不属于泄漏）。
- **坐标系**：WGS84 → 栅格坐标 `[y, x]`（Detroit core bbox + `1024×1024` grid；口径见 `docs/DATA_CONTRACT.md`）。
- **时间分辨率**：1Hz（WorldTrace 标准化后）；`dt=1s` 可以做真实时间尺度的统计（例如速度/加速度/停留）。
- **地图使用方式**：OSM 只作为输入特征（`road_prob/topo/...`）与 soft prior（例如 `L_offroad`），不做训练期 hard cut/masked softmax。
- **训练增强**：ATR + STM（UniTraj 的数据级策略，必须作为独立开关进入消融矩阵；见 `docs/WORDTRACE_UNITRAJ.md`）。

### Legacy（仅用于复现）：深圳 dt30（Phase C）

- 旧设定（dt_fixed=30s）仍保留在仓库中用于复现与对照，但不再作为 Phase D 的任务合同来源。
- 旧口径的命令与审计：见 `docs/HIERARCHICAL_VALIDATION_PROTOCOL.md` 与 `docs/PHASE_C_RESULTS.md`。

---

## 1. 任务定义：Known vs Unknown Destination

| 设定 | 推理时条件 | 适用场景 | 泄漏风险 |
|---|---|---|---|
| **KnownDestination** | 已知 `(o, d, t0)` | 路径生成 / 导航 / 条件生成 | 低（`d` 是输入） |
| **UnknownDestination** | 只知 `(o, t0)` | 轨迹预测 / 异常检测 | 高（若训练使用了 `d`） |

### 1.1 KnownDestination（带观测历史；主线与 legacy 都适用）

本仓库在 KnownDestination 设定下允许两种“输出语义”，两者都必须在实验配置里写死并进入审计：

**（D）Phase D 主线：Macro waypoint 生成（trip-level 决策）**

给定：
- `obs`：历史观测窗口（长度 `H`）
- `o, d`：trip-level 起点/终点
- `t0`：出发时间（建议用 Detroit 本地时区编码；跨城时用城市本地时区）
- `env`：环境特征（例如 `road_prob/topo/poi/landuse`）

学习：

$$P(z \\mid \\mathrm{obs}, o, d, t_0, env)$$

其中：
- `z` 是少量决策点（例如 `z=[wp1, wp2, end]`，每个点是栅格坐标 `[y,x]`）
- 不把 OSM/POI 当真值：它们是条件特征与 soft prior，而不是 hard cut

**（Legacy）窗口级未来段生成（执行层/物理统计对齐）**

在 legacy dt30 或需要对齐旧指标时，最小闭环任务仍可使用“未来段生成”：

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

> **注意**：Phase D 主线的目标是 trip-level 决策（waypoints/走廊/绕路动机）。如果你用 legacy 的“未来段生成”做 Phase D 主线，会很容易回到“Destination Gravity/平均梯度场”的老问题；因此 Phase D 的主要输出应优先用 `z` 表达宏观决策。

---

## 2. 时间与 `dt/vel` 语义

### 2.1 时间戳与时间特征（t0 encoding）

- **统一落盘口径**：所有 `processed_*` 数据中的时间戳统一存为 Unix 秒（`int64`）。
- **WorldTrace（Phase D）**：
  - 输入字段：轨迹 CSV 的 `time`（处理时转换/校验为 Unix 秒）
  - 时间特征按 **城市本地时区**编码（Detroit：`America/Detroit`；跨城时按目标城市时区）
- **Legacy 深圳（Phase C）**：时间特征按 `Asia/Shanghai (UTC+8)` 编码
- **最小时间条件向量（KISS）**：2 维（可作为默认实现，后续可升级）
  - `hour_norm = hour / 23`
  - `weekday_norm = weekday / 6`（Monday=0）

> 若需要更合理的周期编码（`sin/cos` + `is_weekend`），会改变条件维度，需要同步修改模型与重训；必须作为显式开关进入消融矩阵，避免隐性口径漂移。

### 2.2 `dt` 的边界与论文版要求（关键）

- **WorldTrace（Phase D）**：标准化后为 1Hz（`dt=1s`），可直接做真实时间尺度的速度/加速度统计与约束（但仍要审计异常 gap 与时钟问题）。
- **Legacy 深圳（Phase C）**：原始 GPS 采样间隔不固定；若需要严格物理解释，必须重采样到固定 `dt_fixed`（例如 30s）并重训。

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

Phase D（WorldTrace×Detroit）不再把“深圳 HDF5 + nav_field”当作默认落盘形态；我们需要的合同要点是：

- **可复现索引**：manifest（parquet/arrow）+ Detroit 子集切片规则（写入 `docs/DATA_CONTRACT.md`）
- **可复现特征**：OSM/SafeGraph/landuse 等外部特征必须带版本与参数（bbox/grid/sigma/buffer/dilation…）
- **train-only 统计**：任何用于训练的归一化/密度/先验统计（若存在）必须标注 `split=train` 与输入指纹（sha/版本）

推荐产物形态（示例，名称仅作约定，不限定实现）：
- `data/processed_worldtrace_detroit/manifest.parquet`（全局索引，用于筛选/抽样/统计）
- `data/processed_worldtrace_detroit/segments.parquet`（Detroit core 连续段，含 `traj_id/time/lat/lon/is_matched/...`）
- `data/processed_worldtrace_detroit/osm_road_mask.npy` / `osm_dist_to_road_m.npy` / `osm_road_prob.npy`
- `data/processed_worldtrace_detroit/poi_density_*.npy` / `landuse_dom.npy` / `landuse_entropy.npy`

> Legacy（深圳 dt30）相关的 `nav_field.npz`/HDF5 产物合同与命令，统一放在 `docs/HIERARCHICAL_VALIDATION_PROTOCOL.md` 与 `docs/archive/`，避免与 Phase D 口径混淆。

---

## 4. 评估协议（v1）

### 4.1 采样设置（生成模型）

- 生成模型：每个条件采样 `K=20`
- **确定性 L2 回归（Deterministic L2 Regression / SeqBaseline）**：`K=1`（或重复 K 次但 std=0，本质等价）

### 4.2 微观指标（窗口级）

- ADE：平均位移误差
- FDE：终点位移误差
- Fréchet：离散 Fréchet 距离（轨迹形状距离）
- DTW：Dynamic Time Warping 距离（允许时间对齐的形状距离）
- 报告口径（生成模型）：`mean / std / best-of-K`
  - `best-of-K`：对每条样本取 K 条生成里误差最小的那条，再在 batch 上取平均

> **对位说明（论文叙事）**：  
> - **确定性 L2 回归（SeqBaseline）**的优化目标是条件均值轨迹，主看 `ADE_mean/FDE_mean`（`K=1`）。  
> - **生成式模型（Diffusion/Physics/CVAE 等）**主看 `best-of-K`（覆盖潜力上界）以及后续补充的分布指标（如 Energy Score/CRPS），避免把“均值回归器”当作多模态生成的主要竞争对手。

> **收缩问题的处理（重要）**：  
> 若发现生成轨迹的宏观幅度（`Rog/MSD/path_len`）系统性偏小，必须优先使用 **Scale（`vel_scale`）** 做幅度校准（val→test），而不是用采样 temperature/噪声强度去“撑大位移”（会引入抖动且不可控）。  
> `src/training/evaluate.py` 支持 `--vel_scale`；若使用，需在论文中明确校准协议与是否对所有模型一致。

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
  --exp_name physics_v1_strict \
  --seed 0
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
  --num_samples_per_condition 20 \
  --seed 0
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

#### 7.2.1 “异常数据 / inactive 数据”处理口径（避免评估口径混乱）

这里把“会被剔除的异常”与“不会被剔除的 inactive”明确分开（非常关键：否则会误以为 GT 已过滤，导致错误归因）。

- **会被 dt-fixed 产物构建阶段剔除的异常（trajectory-level drop）**：
  - 非单调时间戳（`dt<0`）→ 丢弃该轨迹
  - 去重后仍出现 `dt<=0` → 丢弃该轨迹
  - 存在超大 gap（`max_gap`，默认 300s）→ **丢弃整条轨迹**（保持 trip-level OD 语义一致性）
  - 总时长不足以支撑 `min_length`（默认 10）→ 丢弃该轨迹
  - 事实证据：`data/processed_dt30/resample_meta.json` 会记录 drop 统计；实现代码在 `src/data/build_dt_fixed_dataset.py`（函数 `_resample_one`）。

- **不会被剔除的 inactive（窗口/局部静止）**：
  - “近静止/低位移”的片段不会在数据集层面被删除；`SeqDataset/DiffusionDataset` 只是滑窗切片，不按速度/位移阈值过滤（见 `src/data/datasets_seq.py`、`src/data/datasets_diffusion.py`）。
  - **因此：训练用的 GT 与评估用的 GT 都包含这些窗口**；这也是 macro loss 可能被“低位移窗口稀释”的原因之一（见 `docs/archive/memos/PROFESSOR_UPDATE_BATCH_EPE.md`）。
  - 但在某些统计指标里会做 *metric-level mask*：例如 Turn Angle 统计通常会对 `speed < turn_min_speed` 的 step 做忽略，以避免 “速度≈0 时航向角不稳定” 造成的数值污染（这是指标计算口径，不等价于删数据）。

- **关于 raw 数据的 `status`（是否载客/有效）字段**：
  - 当前 HDF5 产物格式不保存 `status`，训练/评估链路无法按 `status` 过滤（见 `src/data/trajectories.py` 中的字段定义）。
  - 若未来要按 `status` 做“载客/非载客”剔除，必须在 `raw → processed` 阶段把该规则写入产物合同并落地到数据转换脚本中（否则无法复现）。

#### 7.2.2 raw→processed（Passenger Trip）推荐实现（status==1）

若你要从 `data/raw/gps/*.txt` 重新构建更“干净的导航意图”数据集，推荐按 Passenger Trip（`status==1`）抽取：

- 只保留 `status==1`（Passenger Trip），避免把 `status==0` 的 Search Policy 混入导航分布
- `max_gap_s=300`：gap 视为因果断裂，不跨 gap 插值（切段）
- `max_speed_kmh=120`：超速边界视为 GPS 漂移/时间戳错误（切段）

脚本（依赖 `h5py`，无 pandas）：

```bash
python -m src.data.build_passenger_dataset_from_raw_txt \
  --raw_gps_dir data/raw/gps \
  --output_dir data/processed_passenger \
  --keep_status 1 \
  --max_gap_s 300 \
  --max_speed_kmh 120 \
  --min_points 10 \
  --min_od_m 500 \
  --time_zone shanghai
```

> `--time_zone` 说明：本项目的深圳出租车 `data/raw/gps/*.txt` 的 `time` 已确认是北京时间（UTC+8），因此默认使用 `shanghai`。

生成完成后，务必跑 strict(train-only) 产物以避免泄漏并补齐 `data_stats.json/nav_field.npz`：

```bash
python -m src.data.build_strict_products --processed_dir data/processed_passenger --backup
python -m src.utils.sanity_check --data_path data/processed_passenger --strict
```

如果你要进入 Phase B（论文版 dt=30s），再从 passenger 版本生成 dt-fixed 版本：

```bash
python -m src.data.build_dt_fixed_dataset \
  --input_processed_dir data/processed_passenger \
  --output_processed_dir data/processed_passenger_dt30 \
  --dt_fixed 30 \
  --max_gap 300 \
  --min_length 10

python -m src.data.build_strict_products --processed_dir data/processed_passenger_dt30 --backup
python -m src.utils.sanity_check --data_path data/processed_passenger_dt30 --strict --expected_dt 30 --dt_require_constant
```

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

**对比模型集合**（paper-ready）：

| 模型 | 描述 | 物理约束 |
|-----|------|---------|
| Deterministic L2（SeqBaseline） | 确定性序列预测（L2 回归；输出条件均值，`K=1`） | 无 |
| CVAE（baseline） | 条件变分自编码器（多模态生成；与 Diffusion 同类对位） | 无 |
| Data-only Diff | 纯数据扩散生成 | 无 |
| Physics Diff | 物理约束扩散 | nav_field + (可选) macro reg |

**三层评估指标**：

| 层次 | 指标 | 说明 |
|-----|------|-----|
| 微观 | ADE, FDE, Fréchet, DTW（mean/std/best-of-K） | 单条轨迹误差 |
| 中观 | (v2) 路径分布, OD 匹配 | 需要 road-level |
| 宏观 | MSD 曲线, Rog 分布（同时输出 GT 对照） | 物理是否在帮忙 |

**消融实验**：

| 实验 | 目的 |
|-----|------|
| Physics vs Data-only | nav_field 的贡献 |
| w/ vs w/o destination | KnownDest 的影响 |
| dt=30s vs step-based | 重采样的影响 |
| K=5,10,20 | 采样数敏感性 |

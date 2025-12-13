# 数据结构说明（DATA_STRUCTURE）

> [!IMPORTANT]
> 任务定义与实验协议以 `docs/TASK_DEFINITION.md` 为唯一准则；本文档仅描述数据格式，若与其冲突以其为准。

目标：
- 统一所有数据文件的格式、路径和坐标约定；
- 分清 raw / processed / experiment 三个层级；
- 保证任何人只看这个文档就能正确读写数据。

---

## 1. 坐标与时间约定

### 1.1 地理坐标与栅格坐标

- **原始 GPS**：使用经纬度 `(lat, lon)`，外部格式（csv/parquet）
- **模型内部**：使用栅格坐标 `(y, x)`，遵循图像坐标系

```text
 ┌──────────────────────► x (col)
 │
 │  (0,0)
 ▼ y (row)
```

**统一约定：**

| 对象 | 约定 | 说明 |
|-----|------|-----|
| 2D 数组 `field` | `field[y, x]` | 行优先，y 为行索引 |
| 位置向量 | `[y, x]` | 第一维是 y |
| `vel` 向量 | `[vy, vx]` | **步位移**（step displacement），与位置一致，单位 `grid_cell/step` |
| 导航方向 | `[nav_y, nav_x]` | 单位向量 |

> [!IMPORTANT]
> 所有代码中的位置、速度、方向向量**必须**遵循 `[y, x]` 约定。这是为了与图像坐标系、numpy 数组索引保持一致。

### 1.2 时间约定

| 格式 | 用途 |
|-----|------|
| Unix 时间戳（秒，int64） | 数据存储、计算 |
| ISO 字符串（建议带时区） | 日志、可读性展示（例如 `+08:00`） |

**模型输入中的时间特征：**

```python
time_features = {
    "hour_of_day": int,      # 0-23
    "day_of_week": int,      # 0-6 (Monday=0)
    "is_weekend": bool,      # 可选（v1 strict 当前未用）
    "is_holiday": bool,      # 可选
}
```

### 1.3 数据归一化 (Normalization)

> [!WARNING]
> **必须严格执行**：为了保证 Diffusion 模型训练稳定，所有连续值特征必须归一化。

| 特征 | 源范围 (Approx) | 目标范围 | 方法 | 统计量存储 |
|-----|---------------|---------|-----|-----------|
| **Position** `[y, x]` | `[0, H]`, `[0, W]` | `[-1, 1]` | Min-Max (Linear) | `data_config['pos_bounds']` |
| **Vel** `[vy, vx]` | data-dependent | $\approx [-3, 3]$ | Z-Score | `data_config['vel_mean/std']` |
| **Nav** `[ny, nx]` | `[-1, 1]` | `[-1, 1]` | 缩放 (e.g. $\times 1.0$) | None |

**注意事项**：
- 训练集/验证集/测试集**必须使用同一套归一化参数**（通常只用训练集统计量）。
- 评估时必须**反归一化**回到原始物理空间计算指标。

> 若需要物理速度（单位 `grid_cell/second`），在 dt-fixed 数据集上可使用：
> `physical_velocity = vel / dt_fixed`。
>
> `nav_patch` 的 speed 通道会按 `nav_max_speed` 做归一化（见 `data_stats.json.normalization.nav_max_speed`）；若字段缺失则使用默认值（当前为 20.0）。

---

## 2. 目录层级

```text
data/
├── raw/
│   ├── gps/              # 原始车辆 GPS 轨迹
│   └── network/          # 路网 (OSM 等)
├── processed/
│   ├── map_matched/      # 地图匹配后的轨迹
│   ├── trajectories/     # 统一格式的 trip 序列
│   ├── splits/           # train/val/test 切分
│   ├── data_stats.json   # strict: train-only 统计量（含 source）
│   ├── nav_field.npz     # strict: train-only 导航场（含 metadata）
│   ├── fields/           # legacy: 导航场 / 速度场等物理场（可选）
│   └── macro_stats/      # 宏观统计指标（标度律等）
└── experiments/
    └── {exp_name}/       # 各实验的中间结果与评估输出
```

> [!NOTE]
> Phase B（论文版）推荐使用独立目录 `data/processed_dt30/`（dt_fixed=30s），由 `python -m src.data.build_dt_fixed_dataset` 生成，并在该目录下再生成 strict(train-only) 的 `data_stats.json/nav_field.npz`。

**层级说明：**

| 层级 | 内容 | 谁写入 | 谁读取 |
|-----|------|-------|-------|
| `raw/` | 原始数据，不修改 | 外部/手动 | `src/data/preprocess.py` |
| `processed/` | 标准化后的数据 | `src/data/preprocess.py` | 所有训练/评估代码 |
| `experiments/` | 实验结果 | 训练/评估脚本 | 分析/可视化 |

---

## 3. Raw 层：原始数据格式

### 3.1 原始 GPS (`data/raw/gps/*.parquet` 或 `.csv`)

**推荐列结构：**

| 列名 | 类型 | 必需 | 说明 |
|-----|------|-----|------|
| `vehicle_id` | str / int | ✓ | 车辆唯一标识 |
| `timestamp` | int64 | ✓ | Unix 时间戳（秒） |
| `lat` | float64 | ✓ | 纬度 |
| `lon` | float64 | ✓ | 经度 |
| `speed` | float32 | 可选 | 速度（m/s） |
| `heading` | float32 | 可选 | 航向角（度） |

**示例数据：**

```csv
vehicle_id,timestamp,lat,lon,speed,heading
V001,1672531200,31.2304,121.4737,8.5,45.0
V001,1672531260,31.2310,121.4745,10.2,50.0
...
```

### 3.2 路网 (`data/raw/network/{city}.pbf` 或等价格式)

- 原始 OSM/路网文件
- 由 `src/data/preprocess.py` 解析为内部表示（路段、路口）
- 具体格式依赖使用的地图匹配工具

---

## 4. Processed 层：标准化轨迹与物理场

### 4.1 地图匹配后的轨迹

**路径：** `data/processed/map_matched/{city}_mapmatched.parquet`

**列结构：**

| 列名 | 类型 | 说明 |
|-----|------|------|
| `trip_id` | int64 | 同一 trip 的所有点共享同一 ID |
| `vehicle_id` | int64 | 车辆 ID |
| `timestamp` | int64 | Unix 时间戳 |
| `lat`, `lon` | float64 | 原始经纬度 |
| `y`, `x` | float32 | 栅格坐标（投影后） |
| `road_id` | int64 | 匹配到的路段 ID |

> [!NOTE]
> `trip_id` 用于标识一次完整行程。同一 `vehicle_id` 可能有多个 `trip_id`。

### 4.2 统一轨迹文件（HDF5）

**路径：** `data/processed/trajectories/{city}_trajectories.h5`

**采用"扁平 + 指针"结构，支持变长轨迹：**

```text
{city}_trajectories.h5
├── positions     : (N_points, 2) float32   # [y, x] 所有点
├── timestamps    : (N_points,) int64        # 时间戳
├── traj_ptr      : (N_traj+1,) int64        # 指针数组
├── origin_idx    : (N_traj,) int64          # 起点 index
├── dest_idx      : (N_traj,) int64          # 终点 index
└── meta/
    ├── vehicle_id  : (N_traj,) int64
    ├── start_time  : (N_traj,) int64
    └── end_time    : (N_traj,) int64
```

**访问第 i 条轨迹：**

```python
import h5py

with h5py.File("trajectories.h5", "r") as f:
    ptr = f["traj_ptr"][:]
    positions = f["positions"][:]
    
    # 获取第 i 条轨迹
    i = 42
    traj_i_pos = positions[ptr[i]:ptr[i+1]]  # shape: (len_i, 2)
    
    # 在线计算步位移（vel）
    traj_i_vel = np.diff(traj_i_pos, axis=0)  # shape: (len_i-1, 2)
```

**设计说明：**

| 设计选择 | 原因 |
|---------|-----|
| 扁平存储 | 避免 padding，节省空间 |
| 指针数组 | 支持变长，O(1) 访问 |
| 速度在线计算 | 无需额外存储，保证一致性 |

### 4.3 物理场与导航场

#### 4.3.1 v1 strict 导航场（推荐）

**路径：** `data/processed/nav_field.npz`

**结构：**

```text
nav_field.npz
├── direction : (2, H, W) float32  # 平均方向单位向量 [dir_y, dir_x]
├── speed     : (H, W) float32     # 平均步位移模长（step displacement magnitude）
├── count     : (H, W) float32     # 样本数量（置信度）
└── metadata  : object (dict)      # 来源/哈希/dt 等（np.load 需 allow_pickle=True）
```

**加载示例：**

```python
import numpy as np

data = np.load("nav_field.npz", allow_pickle=True)
direction = data["direction"]  # (2, H, W)
speed = data["speed"]          # (H, W)
count = data["count"]          # (H, W)
metadata = data["metadata"].item() if "metadata" in data else None

# 获取某位置的导航方向
y, x = 100, 200
nav_direction = direction[:, y, x]  # [dir_y, dir_x]
```

> [!CAUTION]
> **向量方向一致性警告**：
> 当从 `(lat, lon)` 投影到 `(y, x)` 栅格坐标时，所有**向量特征**（速度、导航方向）必须进行相应的**旋转变换**。
> **推荐做法**：直接在 `(y, x)` 栅格空间上计算速度和平均流场，避免复杂的投影旋转计算。

#### 4.3.2 legacy/兼容导航场（可选）

> 说明：历史版本可能使用 `nav_y/nav_x/speed_mean` 命名；当前代码加载器兼容两种格式，但以 v1 strict 为准。

**路径：** `data/processed/fields/nav_field_baseline.npz`

**结构：**

```text
nav_field_baseline.npz
├── nav_y       : (H, W) float32
├── nav_x       : (H, W) float32
├── speed_mean  : (H, W) float32
└── count       : (H, W) int32
```

#### 4.3.3 目的地相关的导航场（可选，v2）

**路径：** `data/processed/fields/nav_field_dest_{dest_id}.npz`

**结构：**

```text
nav_field_dest_{dest_id}.npz
├── nav_y          : (H, W) float32
├── nav_x          : (H, W) float32
└── distance_field : (H, W) float32   # 到该目的地的栅格距离
```

> [!TIP]
> 这与旧项目中 "每个 sink 一个导航场" 的做法形式相同，只是语义从 "sink" 换成了更通用的 "destination"。

### 4.4 数据集切分

**路径：** `data/processed/splits/`

**文件结构：**

```text
splits/
├── train_ids.npy    # (N_train,) int64 - 训练集轨迹 ID
├── val_ids.npy      # (N_val,) int64   - 验证集轨迹 ID
└── test_ids.npy     # (N_test,) int64  - 测试集轨迹 ID
```

**切分策略建议：**

| 策略 | 适用场景 | 优点 | 缺点 |
|-----|---------|-----|-----|
| 按时间切分 | 评估泛化能力 | 无时间泄漏 | 季节性差异 |
| 按车辆切分 | 评估跨用户泛化 | 更严格 | 可能丢失模式 |
| 随机切分 | 快速实验 | 简单 | 可能有泄漏 |

> [!WARNING]
> **推荐按时间切分**：例如最后一周作为测试集，避免信息泄漏。

### 4.5 宏观统计指标

**路径：** `data/processed/macro_stats/{city}_macro.json`

**结构示例：**

```json
{
  "dataset_info": {
    "city": "shanghai",
    "num_trajectories": 50000,
    "num_points": 2500000,
    "date_range": ["2023-01-01", "2023-06-30"]
  },
  "displacement_distribution": {
    "bins": [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    "hist": [0.05, 0.15, 0.25, 0.30, 0.15, 0.07, 0.03]
  },
  "msd_scaling": {
    "delta_t_grid": [1, 2, 5, 10, 20, 50, 100],
    "msd_values": [0.5, 1.8, 10.2, 35.5, 120.3, 650.2, 2100.5],
    "alpha_fitted": 1.72,
    "r_squared": 0.98
  },
  "radius_of_gyration": {
    "mean": 5.2,
    "std": 3.8,
    "percentiles": {"25": 2.1, "50": 4.3, "75": 7.5, "95": 12.8}
  },
  "step_length": {
    "mean": 0.35,
    "std": 0.42
  }
}
```

**用途：**
- 训练 physics-informed 模型时的目标统计
- 评估时比较生成vs真实的宏观特性

---

## 5. Experiments 层：实验特定输出

**路径：** `data/experiments/{exp_name}/`

**当前实现（v1）结构：**

```text
data/experiments/{exp_name}/
├── last.pt        # 训练保存的权重（baseline: dict；diffusion/physics: state_dict）
├── epoch_*.pt     # baseline 可选保存（每 5 epoch）
├── metrics.json   # evaluate.py 输出（可选）
└── samples.npz    # evaluate.py 输出（可选）
```

> 说明：更完整的实验管理（config 快照、checkpoints/logs 子目录、meso 指标等）建议作为后续工程化增强，但不影响 v1 strict 的可复现闭环。

---

## 6. 快速参考

### 6.1 文件格式速查表

| 数据类型 | 格式 | 路径模式 |
|---------|------|----------|
| 原始 GPS | parquet/csv | `data/raw/gps/*.parquet` |
| 地图匹配结果 | parquet | `data/processed/map_matched/*.parquet` |
| 统一轨迹 | HDF5 | `data/processed/trajectories/*.h5` |
| 导航场 | npz | `data/processed/nav_field.npz`（推荐）或 `data/processed/fields/*.npz`（legacy） |
| 数据切分 | npy | `data/processed/splits/*.npy` |
| 宏观统计 | json | `data/processed/macro_stats/*.json` |
| 模型权重 | pt | `data/experiments/*/*.pt` |
| 评估结果 | json/npz | `data/experiments/*/metrics.json`、`data/experiments/*/samples.npz` |

### 6.2 坐标约定速查

```python
# ✅ 正确
pos = [y, x]                # 位置向量
vel = [vy, vx]              # 速度向量
field_value = field[y, x]   # 2D 数组访问
nav = [nav_y, nav_x]        # 导航方向

# ❌ 错误
pos = [x, y]                # 顺序错误！
field_value = field[x, y]   # 索引顺序错误！
```

### 6.3 时间约定速查

```python
# Unix 时间戳（存储用）
timestamp = 1672531200  # int64, 秒

# 转换为可读格式（展示用）
from datetime import datetime, timedelta, timezone
tz = timezone(timedelta(hours=8))  # Asia/Shanghai
dt = datetime.fromtimestamp(timestamp, tz=tz)
iso_str = dt.isoformat()  # "2023-01-01T08:00:00+08:00"

# 提取时间特征（模型输入用）
hour = dt.hour            # 0-23
day_of_week = dt.weekday()  # 0-6
```

---

## 7. 与旧工程的关系说明

**保留的设计：**
1. **统一的坐标约定**：`[y, x]` 和 `(H, W)` 栅格
2. **导航场结构**：`direction` + `(2, H, W)`（内部约定 `[y, x]`）
3. **三层数据分离**：raw / processed / experiment

**不再保留的设计：**
- 多阶段流水线（Phase 1-4）
- 复杂的阶段性生成再学习

> [!NOTE]
> 本项目采用简化的数据处理流程，专注于方法论对比，而非复杂的工程流水线。

---

*最后更新：2025-12-09*

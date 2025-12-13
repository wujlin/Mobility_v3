# 开发进度追踪

> [!IMPORTANT]
> 任务定义与实验协议以 `docs/TASK_DEFINITION.md` 为唯一准则；本文件用于进度记录，若与其冲突以其为准。

> **项目**：物理约束扩散模型 - 轨迹生成  
> **数据**：深圳出租车 GPS（2011/04/18-26）  
> **更新**：2025-12-13

---

## 总览

| Phase | 任务 | 状态 |
|-------|------|------|
| 1 | 数据预处理 | ✅ 完成 |
| 2 | 全量处理 | ⏳ 进行中 |
| 3 | 模型训练 | ✅ 代码就绪，待真实数据训练 |
| 4 | 评估验证 | ✅ 代码就绪，待真实数据验证 |

---

## Phase 1: 数据预处理 ✅

### 1.1 GPS 读取 ✅

| 项 | 值 |
|---|---|
| 路径 | `data/raw/gps/*.txt` |
| 格式 | 车牌, 时间, 经度, 纬度, 状态, 速度, 方向 |
| 文件数 | 1185 |
| 实现 | `src/data/raw_io.py` |

### 1.2 网格配置 ✅

```python
H = 400, W = 800
lat: [22.45, 22.85]
lon: [113.75, 114.65]
```

### 1.3 行程切分 ✅

基于状态字段：0=空车, 1=载客
- 0→1: trip 开始
- 1→0: trip 结束

### 1.4 归一化 ✅

| 特征 | 方法 |
|-----|-----|
| pos | Min-Max → [-1, 1] |
| vel | Z-Score |

### 1.5 测试结果（20 文件）

```
轨迹数: 2493
总点数: 70645
车辆数: 20
时间: 2011-04-18 ~ 2011-04-26
轨迹长度: min=10, max=294, mean=28.3
```

---

## Phase 2: 全量处理 ⏳

**命令**：
```bash
python -m src.data.run_preprocess
```

**进度**：运行中...

---

## Phase 3: 模型训练 ✅ 代码就绪

**单元测试全部通过**：
```
test_model_seq.py      ✅ SeqBaseline Test Passed
test_model_diffusion.py ✅ DiffusionModel Test Passed  
test_model_physics.py   ✅ PhysicsModel Test Passed
```

| 模型 | 文件 | 状态 |
|-----|-----|-----|
| Seq Baseline | `train_baseline.py` | ✅ 代码就绪 |
| Diffusion | `train_diffusion.py` | ✅ 代码就绪 |
| Physics | `physics_condition_diffusion.py` | ✅ 代码就绪 |

**下一步**：全量数据处理完成后，运行训练：
```bash
# 0) 生成 strict(train-only) 数据产物（无泄漏）
python -m src.data.build_strict_products --processed_dir data/processed
python -m src.utils.sanity_check --data_path data/processed --strict

# Baseline
python -m src.training.train_baseline --data_path data/processed/trajectories/shenzhen_trajectories.h5 --split train

# Diffusion
python -m src.training.train_diffusion --model_type diffusion --data_path data/processed/trajectories/shenzhen_trajectories.h5 --split train

# Physics
python -m src.training.train_diffusion --model_type physics --data_path data/processed/trajectories/shenzhen_trajectories.h5 --nav_file data/processed/nav_field.npz --split train
```

---

## Phase 4: 评估 ✅ 代码就绪

| 层次 | 指标 | 实现 |
|-----|-----|-----|
| 微观 | ADE, FDE, MSE_k | `micro_metrics.py` ✅ |
| 宏观 | MSD, Rog | `macro_metrics.py` ✅ |

---

## Phase B（论文版）：dt=30s 重采样 ✅ 代码就绪

> 论文版必须 dt-fixed（见 `docs/TASK_DEFINITION.md`），并重新生成 train-only 数据产物以避免任何泄漏。

```bash
# 1) 生成 dt-fixed 数据集（输出到独立目录，避免覆盖 Phase A）
python -m src.data.build_dt_fixed_dataset \
  --input_processed_dir data/processed \
  --output_processed_dir data/processed_dt30 \
  --dt_fixed 30 \
  --max_gap 300 \
  --min_length 10

# 2) 生成 strict(train-only) 数据产物（无泄漏）
python -m src.data.build_strict_products --processed_dir data/processed_dt30 --backup
python -m src.utils.sanity_check --data_path data/processed_dt30 --strict --expected_dt 30 --dt_require_constant
```

## 输出文件

```
data/processed/
├── trajectories/shenzhen_trajectories.h5
├── splits/{train,val,test}_ids.npy
├── data_stats.json
└── nav_field.npz
```

---

## 日志

| 日期 | 事件 |
|-----|-----|
| 2025-12-11 | Phase 1 完成，20 文件测试通过 |
| 2025-12-11 | GitHub 仓库创建，代码推送 |
| 2025-12-11 | Phase 2 全量处理启动 |
| 2025-12-11 | Phase 3/4 单元测试验证通过（代码就绪）|
| 2025-12-13 | strict(train-only) 数据产物生成器与 sanity_check 完成；支持 dt-fixed(30s) 论文版数据集生成闭环 |

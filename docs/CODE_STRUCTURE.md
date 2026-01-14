# 代码结构说明（CODE_STRUCTURE）

> [!IMPORTANT]
> 任务定义与实验协议以 `docs/TASK_DEFINITION.md` 为唯一准则；本文档描述代码组织与接口，若与其冲突以其为准。

目标：统一项目的代码组织方式和核心接口，保证：
- 所有模型共享同一套数据接口；
- 便于做「序列预测 vs 生成模型 vs 物理约束生成」的对比；
- 调试时能快速定位是哪一层出问题（数据 / 模型 / 评估）。

---

## 0. 当前两条工作线（避免新 PI 误解）

本仓库目前同时保留两条“可运行但目的不同”的代码主线：

1) **ICML 2026 Route Generation（当前优先级最高）**  
   - 写作入口：`essay_icml_cascadetraj/main.tex`（当前主线；`essay_population/main.tex` 为备份/对照稿）  
   - 技术主线：segment-level route generation + road-graph 上的结构化决策（waypoint AR）+ A\* 连接（可选再接 continuous execution）。

2) **Paper-2：Rupture/Avoidance field（非当前 ICML 交付物）**  
   - 写作入口：`essay/main.tex`  
   - 这部分文档与脚本仍保留，但不应与 routegen 的实验口径/指标混用。

下面先给 routegen 这条线一个“能直接跑”的代码地图；原先 Phase D（avoidance）内容后移保留。

---

## 0.1 RouteGen（ICML 2026）代码地图（最常用入口）

### 数据与图结构（road graph / GT graph paths）

- 生成/加载 road graph：`src/data/road_graph/*road_graph*.py`
- GT route（segment-level）→ graph node sequence：`src/data/road_graph/dump_graph_paths_from_routes_npz.py`
- 候选覆盖诊断（说明为什么不用 K-shortest+classify）：`src/data/road_graph/gate_candidate_paths_from_routes_npz.py`、`src/data/road_graph/diagnose_candidate_coverage.py`
- 语义信息量 Gate（time+tier 是否 informative）：`src/data/road_graph/gate_semantic_informativeness_cluster.py`

### 决策层（Waypoint AR：少步数、避免 600-step 累积误差）

- GT graph path → 固定 K waypoint：`src/data/road_graph/dump_waypoints_from_paths_graph_npz.py`
- 模型：`src/models/road_graph/ar_waypoint_bins.py`
- 训练：`src/training/train_graph_ar_waypoint_bins.py`

### 执行层（A* 连接 + 可视化/指标）

- 采样 waypoint → A* 连接成 corridor path，并输出 `best-of-K Jaccard + success_rate`：  
  `src/training/sample_graph_ar_waypoints_astar.py`

> 说明：A* 目前是“最小可行执行层”，确保图路径合法；continuous execution（diffusion/flow）属于后续增强，不是当前 gate 的前置条件。

---

## 0.2 Rupture/Avoidance（Paper-2）原主线（保留）

Phase D（avoidance）的目标不是“刷一个更低 ADE”，而是构造 **Behavioral Reference Frame** 并在 Detroit 上产出 **Behavioral Avoidance Field**。因此代码的主线闭环是：

1) **WorldTrace 子集与连续段**（多进程 / IO 为主瓶颈）  
   - `src/data/worldtrace/build_manifest.py`：从 `Meta.zip` 建 manifest（全局索引）
   - `src/data/worldtrace/build_detroit_segments.py`：按 bbox 抽取 city core 连续段（尽管文件名包含 detroit，但脚本支持 `--bbox/--grid_*`）

2) **Geo-context 特征（全部可开关、可审计）**  
   - OSM soft prior（road\_prob）：`src/data/osm/build_osm_road_prob.py`
   - POI 栅格（SafeGraph）：`src/data/safegraph/build_poi_rasters.py`
   - Wayback 遥感瓦片：`src/data/wayback/download_wayback_tiles.py`
   - Census/ACS 外部指标（tract）：`src/data/census/*.py`

3) **决策-执行生成器（作为“参照系”而非终点）**  
   - Macro planner（AR waypoint；Phase D 主线将使用 soft prior，不做 hard cut）：`src/training/train_macro_*` + `src/models/macro/`
   - Micro executor（确定性执行）：复用 `src/training/train_baseline.py` 或后续单独脚本（按需要）

4) **残差空间化与验证**  
   - 标量审计（city story / detour 分布）：`src/evaluation/city_story_analysis.py`
   - G1/G2 审计与诊断脚本：`src/evaluation/*.py`
   - Avoidance field（回避场）构造：`src/evaluation/build_avoidance_field.py`（输入：`expected_segments.parquet` vs `observed_segments.parquet`；输出：`avoidance_log_ratio.npy` 等）

> 说明：Phase D 的外置数据根目录建议用 `$RAW_ROOT`（不进 git），仓库内只放代码与小体量 JSON/PNG；见 `docs/DATA_STRUCTURE.md`。

## 1. 目录结构

项目根目录（建议）：

```text
project/
├── data/                 # 只放数据（见 DATA_STRUCTURE）
├── src/
│   ├── config/           # 配置（yaml/json），实验参数
│   ├── data/             # 数据处理 & Dataset
│   ├── features/         # 物理场、统计物理特征
│   ├── models/           # 各类模型（序列预测 / Diffusion / 物理约束）
│   ├── training/         # 训练脚本（高层逻辑）
│   ├── evaluation/       # 评估与可视化
│   └── utils/            # 通用工具（日志、坐标变换等）
└── docs/
    ├── CODE_STRUCTURE.md
    └── DATA_STRUCTURE.md
```

---

## 2. 任务与模块对应关系

核心任务：学习**条件轨迹分布**

$$P(\tau \mid o, d, t_0, \text{env})$$

并比较三类模型：

1. 纯数据 **序列预测模型**（如 RNN/Transformer）
2. 纯数据 **轨迹扩散生成模型**（Diffusion）
3. **物理约束扩散生成模型**（PDE/标度律 作为先验）

在代码层面，对应三个子模块：

| 模型类型 | 代码位置 | 说明 |
|---------|---------|-----|
| 序列预测 baseline | `src/models/seq/` | RNN/Transformer |
| Data-only 轨迹扩散 | `src/models/diffusion/` | 1D-UNet + DDPM |
| Physics-informed 扩散 | `src/models/physics/` | PDE residual + 宏观约束 |

---

## 3. 核心模块设计

### 3.1 `src/data/` – 数据加载与 Dataset

**职责：**

1. Phase D：从外置 `$RAW_ROOT/` 读取 WorldTrace/OSM/SafeGraph/Wayback/Census 等多源数据（不进 git）
2. Legacy（深圳 dt30，仅复现）：从 `legacy/shenzhen/data/raw/` 构建 `legacy/shenzhen/data/processed*`
3. 提供统一的 PyTorch Dataset 接口

**推荐子结构：**

```text
src/data/
├── raw_io.py              # 读取 raw GPS / 路网
├── preprocess.py          # map-matching、trip 切分、坐标变换
├── trajectories.py        # Trajectory 对象 & 操作
├── datasets_seq.py        # 序列预测 Dataset
└── datasets_diffusion.py  # 扩散生成 Dataset
```

Phase D 新增的数据模块（外置数据根目录 + 可复现产物）：

```text
src/data/
├── worldtrace/             # WorldTrace manifest/segments（多进程抽取）
├── osm/                    # OSM road_mask/dist/road_prob
├── safegraph/              # SafeGraph POI 栅格化（vintage + active 规则）
├── wayback/                # ArcGIS Wayback 遥感瓦片下载
└── census/                 # ACS 指标 + TIGER 边界（tract-level）
```

**统一 sample 结构：**

#### 序列预测 Dataset 返回格式

```python
{
    "obs":        Tensor[H, 4],   # 历史 H 步 [pos, vel]
    "target_pos": Tensor[F, 2],   # 未来 F 步位置（归一化）
    "target_vel": Tensor[F, 2],   # 未来 F 步步位移（归一化）
    "cond":       Tensor[6],      # 条件向量 [hour, weekday, o_y, o_x, d_y, d_x]
    "meta":       dict(...)       # 轨迹索引等（调试用）
}
```

#### 扩散生成 Dataset 返回格式

```python
{
    "obs":    Tensor[H, 4],   # 历史 H 步 [pos, vel]
    "action": Tensor[F, 2],   # 未来 F 步步位移（生成目标）
    "cond":   Tensor[6],      # 条件向量
    # physics 模型额外输入（可选）
    "nav_patch": Tensor[3, K, K]
}
```

> [!NOTE]
> `obs` 里的位置/速度/导航方向都采用 `(y, x)` / `[vy, vx]` 约定，与坐标系统保持一致，避免混乱。

---

### 3.2 `src/features/` – 物理场与统计特征

**职责：**
把"统计物理 / PDE 的知识"显式编码成可用的特征或约束。

**推荐子结构：**

```text
src/features/
├── nav_field.py      # 从真实轨迹估计导航/速度场 (nav_y, nav_x)
├── physics_pde.py    # 简单 PDE / drift 模型 (可选)
└── macro_stats.py    # 标度律等宏观统计指标的计算
```

**关键输出：**

| 输出 | 格式 | 用途 |
|-----|------|-----|
| `nav_field` | `(2, H, W)` 方向场 `[nav_y, nav_x]` | 给模型提供"物理方向"条件 |
| `macro_stats` | JSON/dict | MSD 幂律指数等，用于训练/评估时检查宏观一致性 |

---

### 3.3 `src/models/` – 模型接口与具体实现

**统一基类接口：**

```python
class BaseTrajectoryModel(nn.Module):
    """所有轨迹模型的基类，定义统一接口"""
    
    def forward(self, obs: Tensor, cond: Tensor) -> Tensor:
        """用于训练的前向：返回下一步或未来序列的预测/噪声预测。"""
        raise NotImplementedError

    def sample_trajectory(
        self, 
        obs: Tensor, 
        cond: Tensor, 
        horizon: int, 
        **kwargs
    ) -> Tensor:
        """给定历史 obs 和条件 cond，生成未来 horizon 步轨迹。
        
        Returns:
            Tensor[B, horizon, 2]: 生成的轨迹（位置或速度序列）
        """
        raise NotImplementedError
```

#### 3.3.1 序列预测模型 `src/models/seq/`

```text
src/models/seq/
├── seq_baseline.py    # Deterministic L2（LSTM encoder-decoder）
└── seq_cvae.py        # CVAE baseline（多模态生成）
```

**核心类：**

```python
class SeqBaseline(BaseTrajectoryModel):
    """Deterministic L2 Regression（确定性序列预测）"""
    
    def forward(self, obs, cond):
        # 返回预测的下一步或未来 F 步
        ...
    
    def sample_trajectory(self, obs, cond, horizon, **kwargs):
        # 自回归 rollout
        ...
```

```python
class SeqCVAE(BaseTrajectoryModel):
    """CVAE baseline（多模态生成，对位 Diffusion/Physics）"""
```

#### 3.3.2 轨迹扩散模型 `src/models/diffusion/`

```text
src/models/diffusion/
├── __init__.py
├── unet1d.py          # 1D UNet（时间维卷积）
├── scheduler.py       # DDPM 调度器
└── diffusion_model.py # UNet + scheduler（训练/采样）
```

**核心类：**

```python
class DiffusionTrajectoryModel(BaseTrajectoryModel):
    """Data-only 轨迹扩散模型"""
    
    def forward(self, obs, cond, target=None):
        # 返回 diffusion loss（以 future vel 为 target）
        ...
    
    def sample_trajectory(self, obs, cond, horizon, num_steps=50, cfg_scale=1.0):
        # DDPM/DDIM 采样
        ...
```

#### 3.3.3 物理约束扩散模型 `src/models/physics/`

```text
src/models/physics/
├── __init__.py
├── cnn_encoder.py                 # nav_patch 编码器
├── macro_regularizer.py           # 宏观统计正则项（可选）
└── physics_condition_diffusion.py # 物理条件扩散（nav_patch 作为条件）
```

**核心类：**

```python
class PhysicsConditionDiffusion(BaseTrajectoryModel):
    """物理约束扩散模型：Nav Field 作为 Condition 输入"""
    
    def __init__(self, nav_field, ...):
        self.nav_encoder = CNNEncoder(...)  # 处理局部 Nav Patch
        self.diffusion = DiffusionTrajectoryModel(...)
        ...
    
    def get_nav_patch(self, current_pos):
        """从全局导航场中 Crop 出以 current_pos 为中心的 Patch"""
        ...
    
    def forward(self, obs, cond, target=None, nav_patch=None):
        # 以 nav_patch 编码后拼接到 cond，再计算 diffusion loss
        ...
    
    def sample_trajectory(self, obs, cond, horizon, ...):
        # 同样提取 nav patch 作为 condition 进行采样
        ...

---

### 3.4 `src/training/` – 训练入口

**当前实现：**

```text
src/training/
├── evaluate.py           # 评估入口（支持 K 采样）
├── train_baseline.py     # 训练 Deterministic L2（SeqBaseline）
├── train_cvae.py         # 训练 CVAE baseline
└── train_diffusion.py    # 训练 diffusion / physics（通过 --model_type）
```

Phase D 的主线训练脚本（Macro/AR 方向）：

```text
src/training/
├── train_macro_hardsupport.py        # legacy: hard support（输出空间裁剪）
├── train_macro_hardsupport_ar.py     # legacy: hard support + AR
└── train_macro_diffusion.py          # macro diffusion（Phase D 将作为“多模态意图层”候选；需 gated）
```

> 注：Phase D 已明确“不把 hard support 当能力”。`src/models/macro/*hardsupport*` 作为历史上界/诊断基线保留；主线将实现 soft-prior 版本的 AR planner（同目录下新增文件），并通过消融审计避免把外部地图当真值。

**每个脚本的职责：**

1. 解析配置（数据路径、模型超参、训练参数）
2. 构建 Dataset / DataLoader
3. 构建 model + optimizer + scheduler
4. 调用统一训练 loop

**示例用法：**

```bash
# Legacy（深圳）：Deterministic L2（按 split）
python -m src.training.train_baseline \
  --data_path legacy/shenzhen/data/processed/trajectories/shenzhen_trajectories.h5 \
  --split train \
  --exp_name baseline_v1_strict

# Legacy（深圳）：CVAE baseline（按 split）
python -m src.training.train_cvae \
  --data_path legacy/shenzhen/data/processed/trajectories/shenzhen_trajectories.h5 \
  --split train \
  --exp_name cvae_v1_strict

# Legacy（深圳）：Data-only Diffusion（按 split）
python -m src.training.train_diffusion \
  --model_type diffusion \
  --data_path legacy/shenzhen/data/processed/trajectories/shenzhen_trajectories.h5 \
  --split train \
  --exp_name diff_v1_strict

# Legacy（深圳）：Physics Diffusion（按 split + nav_field）
python -m src.training.train_diffusion \
  --model_type physics \
  --data_path legacy/shenzhen/data/processed/trajectories/shenzhen_trajectories.h5 \
  --nav_file legacy/shenzhen/data/processed/nav_field.npz \
  --split train \
  --exp_name physics_v1_strict
```

---

### 3.5 `src/evaluation/` – 评估与可视化

当前实现：

```text
src/evaluation/
├── micro_metrics.py  # ADE/FDE/Fréchet/DTW
└── macro_metrics.py  # MSD/Rog（step-based；论文版需 dt_fixed）
```

评估入口使用 `src/training/evaluate.py`（支持 split + K 采样）：

```bash
python -m src.training.evaluate \
  --exp_name legacy_phys_dt30_eval \
  --model_type physics \
  --data_path legacy/shenzhen/data/processed/trajectories/shenzhen_trajectories.h5 \
  --checkpoint legacy/shenzhen/data/experiments/phys_dt30_rog_h128_b1024_lr1e-3_e20_s0/last.pt \
  --nav_file legacy/shenzhen/data/processed/nav_field.npz \
  --split test \
  --num_samples_per_condition 20
```

---

## 4. 公共工具与约定

### 4.1 坐标与向量约定

保持统一以免混乱：

| 对象 | 约定 | 示例 |
|-----|------|-----|
| 2D 栅格字段 | `field[y, x]` | `nav_field[:, y, x]` |
| 位置向量 | `[y, x]` | `pos = [10.5, 20.3]` |
| 速度向量 | `[vy, vx]` | `vel = [0.5, 1.2]` |
| 导航方向 | `[nav_y, nav_x]` | 单位向量 |

**`src/utils/coords.py` 建议实现：**

```python
def latlon_to_grid(lat, lon, grid_config):
    """经纬度转栅格坐标 (y, x)"""
    ...

def grid_to_latlon(y, x, grid_config):
    """栅格坐标转经纬度"""
    ...

def normalize_direction(vec):
    """向量归一化为单位向量"""
    ...
```

### 4.2 配置与实验管理

**配置文件结构（YAML）：**

```yaml
# configs/diffusion_dataonly.yaml
experiment:
  name: "diffusion_baseline_v1"
  seed: 42

data:
  trajectory_file: "legacy/shenzhen/data/processed/trajectories/shenzhen_trajectories.h5"  # legacy 示例
  nav_field_file: "legacy/shenzhen/data/processed/nav_field.npz"                           # legacy 示例
  history_len: 4
  future_len: 16
  batch_size: 256

model:
  type: "diffusion"
  hidden_dim: 128
  num_layers: 4
  diffusion_steps: 100

training:
  epochs: 100
  lr: 1e-4
  weight_decay: 1e-5
  
evaluation:
  eval_every: 5
  num_samples: 10
```

**约定：**
- 每次实验写一个配置文件
- 训练脚本读取 config，不在代码里硬编码路径/超参
- 默认实验结果保存到 `data/experiments/{exp_name}/`（建议软链到外置 `$RAW_ROOT/experiments/`）；legacy 深圳产物封存在 `legacy/shenzhen/data/experiments/{exp_name}/`

---

## 5. 开发流程建议

### 5.1 推荐开发顺序

```mermaid
graph LR
    A[1. src/data/] --> B[2. src/models/seq/]
    B --> C[3. src/evaluation/micro]
    C --> D[4. src/models/diffusion/]
    D --> E[5. src/features/]
    E --> F[6. src/models/physics/]
    F --> G[7. 完整评估]
```

1. **先完成 `src/data/`**：确保数据加载正确
2. **实现 seq baseline**：快速验证 pipeline
3. **实现微观评估**：确保评估代码可用
4. **实现 diffusion**：对比生成 vs 预测
5. **实现物理特征**：导航场、PDE drift
6. **实现 physics-informed**：核心贡献
7. **完整三层评估**：验证方法论

### 5.2 测试建议

```text
tests/
├── test_data_loading.py      # 测试数据加载
├── test_model_forward.py     # 测试模型前向传播
├── test_sampling.py          # 测试采样生成
└── test_metrics.py           # 测试评估指标计算
```

---

*最后更新：2025-12-09*

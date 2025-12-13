# 实验 Checklist：物理约束轨迹扩散模型

> [!IMPORTANT]
> 任务定义与实验协议以 `docs/TASK_DEFINITION.md` 为唯一准则；本 checklist 仅作执行清单，若与其冲突以其为准。

> **总目标**：从同一套真实车辆 GPS 轨迹出发，在同一个任务（条件轨迹分布）上，对比"纯数据序列预测 → 纯数据生成 → 引入物理约束的生成"，用微观 / 中观 / 宏观三个层级的指标，系统回答"物理约束 + 生成模型"到底带来了什么。

---

## 0. 总目标（先在脑子里锁死）

- [ ] 统一建模对象：学的是**条件轨迹分布** $P(\tau \mid o,d,t_0,\text{env})$，而不是单点预测
- [ ] 对比三类模型能力：
  1. 纯数据的**序列预测**（RNN/Transformer）
  2. 纯数据的**轨迹扩散生成**（Diffusion）
  3. 加入物理先验的**物理约束扩散生成**

---

## 1. 数据准备层

### 1.1 GPS 原始数据 → 路网对齐

- [ ] 选定一个城市/区域，整理车辆 GPS 数据（至少包括 time, lat, lon, vehicle_id）
- [ ] 进行 map-matching：
  - [ ] 将轨迹投影到道路网络（得到 road_id, offset）
- [ ] 选择统一坐标系：
  - [ ] 例如投影到栅格 `(y, x)` 坐标系（和现有工程保持 `[y, x]` 约定）

### 1.2 Trip 切分 & 条件变量

- [ ] 定义 trip：同一车辆在较长停留间隔之间的一段连续行驶
- [ ] 为每个 trip 提取条件：
  - [ ] 起点位置 `o`（可为道路 / 区域 ID）
  - [ ] 终点位置 `d`（最后停留位置）
  - [ ] 出发时间 `t0`（小时 + 周几）
  - [ ] 可选：简单环境特征（如当前全局流量 level）

### 1.3 序列数据格式（对齐现有 Phase 4 结构）

- [ ] 对每个 trip，构建时间序列：
  - [ ] `pos[t] = [y, x]`
  - [ ] `vel[t] = pos[t] - pos[t-1]`（注意与你现在仿真数据的定义一致）
- [ ] （论文版）将轨迹重采样到固定 `dt_fixed`（例如 30s），并写死 gap/去重规则，避免 dt 语义不清
- [ ] 定义滑动窗口：
  - [ ] history 长度 H（例如 2–4 步）
  - [ ] future 长度 F（例如 8–16 步）
- [ ] 为每个窗口生成样本：
  - [ ] obs: `[(pos, vel, nav)_t]_{t=t0..t0+H-1}`（nav 暂时可以先留空或用简单方向场）
  - [ ] action: `[vel_{t0+H..t0+H+F-1}]`

### 1.4 数据归一化 (Normalization) & 校验

- [ ] **实现归一化逻辑**：
  - [ ] POS: MinMax 到 [-1, 1] (基于此时数据集的 bbox)
  - [ ] VEL: Z-Score (Mean=0, Std=1)
  - [ ] NAV: Scale 到 [-1, 1] 左右
- [ ] **向量方向校验**：
  - [ ] 随机抽取 100 条轨迹，计算 `cos_sim(nav_direction, true_velocity)`
  - [ ] 确保平均相似度 > 0，否则检查坐标变换旋转矩阵

---

## 2. 模型 A：纯数据序列预测 Baseline

### 2.1 模型设计

- [ ] 选择一个简单但可靠的 baseline：
  - [ ] RNN/LSTM 或 Transformer encoder（时间维度）
- [ ] 输入：
  - [ ] 历史序列 obs（H 步）+ 条件向量 c = (o, d, t0, env)
- [ ] 输出：
  - [ ] 单步：下一步 `pos_{t+1}` 或 `vel_{t+1}`
  - [ ] 多步：未来 F 步的位置/速度序列

### 2.2 训练

- [ ] 损失：
  - [ ] 连续坐标：MSE（L2）
  - [ ] 若离散栅格 / 道路 ID：交叉熵
- [ ] 划分训练 / 验证 / 测试（按时间切分，避免泄漏）

### 2.3 评估（给后面模型当对照）

- [ ] 一步预测：
  - [ ] Top-k accuracy（如果是离散）
  - [ ] 平均 / 中位距离误差
- [ ] 多步 rollout：
  - [ ] 以模型预测作为下一步历史，滚动预测 H+F 步，画/统计误差随步数变化

---

## 3. 模型 B：纯数据轨迹扩散生成（Data-only Diffusion）

### 3.1 模型与数据对接

- [ ] 使用你现有的 1D-UNet + DDPM 框架：
  - [ ] act 序列 = future F 步 velocity（形状 `(F, 2)`）
  - [ ] obs 展平为全局条件向量（`history * feature_dim`）
- [ ] 条件：
  - [ ] obs（历史局部状态）
  - [ ] c = (o, d, t0, env) 作为额外 embedding 拼在 cond 上

### 3.2 训练

- [ ] 标准扩散训练：
  - [ ] 在真实轨迹 action 上加噪声，训练 denoise 预测 ε 的 MSE loss
- [ ] 使用你已经验证有效的条件注入方式（AdaLN/FiLM + CFG 结构，而不是简单加法）

### 3.3 推理

- [ ] 给定 (obs, c)，采样多条未来 F 步速度序列
- [ ] 将这些速度 roll 到位置轨迹上

### 3.4 评估

- [ ] 微观：
  - [ ] 与真实 future 序列对比：Fréchet / DTW / 平均距离误差（取多条 sample 的 best of K 或平均）
- [ ] 中观：
  - [ ] 对同一 (o,d,t0) 条件生成多条 sample，看真实 vs 生成的：
    - [ ] 路径选择分布（经过哪些道路）
    - [ ] 行程时间分布
- [ ] 宏观：
  - [ ] 基于大量生成轨迹，看位移长度分布、活动半径分布是否接近真实数据

---

## 4. 模型 C：物理约束扩散生成（Physics-Informed Diffusion）

> **方法论重点**：这是核心贡献所在

### 4.1 从真实轨迹估计"物理场"

- [ ] 基于 GPS + 路网，估计一个"经验速度 / 导航场"：
  - [ ] 在每个道路 / 栅格点上，统计平均速度向量（方向 + 模长）
  - [ ] 将其存成类似当前的 `nav_field` 格式 `(2, H, W)`，与坐标约定一致
- [ ] 可选：用简单 PDE / 势场平滑这个经验场（比如解一个泊松方程得到光滑势场）

### 4.2 局部层：Nav Field 作为 Condition

> **变更**：放弃 Residual Learning，改用 Condition Learning

- [ ] **实现 Nav Patch 提取器**：
  - [ ] 给定当前位置 $pos$，从全局 Nav Field Crop 出 $K \times K$ 的局部区域
- [ ] **修改 Diffusion 模型输入**：
  - [ ] 增加一个 CNN Encoder 分支，处理 Nav Patch
  - [ ] 将提取的 Nav Embedding 拼接到 global condition 中
- [ ] **训练目标**：
  - [ ] 保持直接预测 velocity (与 Model B 一致)
  - [ ] 让模型通过 Attention/Concat 机制自动利用 Nav 信息

### 4.3 宏观层：统计物理约束（可先简化）

- [ ] 从真实 GPS 上，先离线估计几个简单的宏观指标（选一两个就够）：
  - [ ] 例如：不同时间尺度 Δt 下的 MSD vs Δt 的幂律指数
  - [ ] 不同距离城中心 r 的波动强度 vs r 的衰减型
- [ ] 在训练 physics-informed diffusion 时，周期性地：
  - [ ] 从模型当前参数生成一批轨迹样本
  - [ ] 估计对应的宏观指标 α_gen
  - [ ] 加一个简单正则：$\mathcal{L}_\text{macro} = \sum (\alpha_\text{gen} - \alpha_\text{data})^2$
  - [ ] 总损失：`L = L_diff + λ * L_macro`

> [!TIP]
> **实操建议**：先只做 **"PDE residual + nav_field condition"**，宏观约束可以作为后续扩展（避免一开始过重）。

### 4.4 评估（对比 Data-only Diffusion）

> [!CAUTION]
> **统一评估标准**：所有输出（无论是 pos 还是 vel）都必须积分/转换为**位置序列**后，再计算下列指标。

- [ ] 重复模型 B 的评估所有指标
- [ ] 特别关注：
  - [ ] 中观：路径效率（到达率、平均绕路率、平均速度）
  - [ ] 宏观：统计物理指标是否明显更接近真实

> [!NOTE]
> `src/training/evaluate.py` 会输出 ADE/FDE + Fréchet/DTW（以及 MSD/Rog）；生成模型统一按 `mean/std/best-of-K` 聚合。

---

## 5. 统一的对比与消融（核心表格）

最后需要有一张"谁比谁强"的结构清楚的对比。

### 5.1 模型版本（纵向）

- [ ] Baseline-A：RNN/Transformer（预测）
- [ ] Baseline-B：Data-only Diffusion（生成）
- [ ] Model-C1：Physics-informed Diffusion（Nav Condition）
- [ ] Model-C2：Physics-informed + Macro regularizer（如果做了的话）

### 5.2 评价维度（横向）

- [ ] 微观：
  - [ ] 单步 / 多步误差（距离、cos 相似度）
- [ ] 中观：
  - [ ] 行程时间分布、路径长度/绕路率、到达率
- [ ] 宏观：
  - [ ] 位移分布、活动半径分布、选定的标度律指标

---

## 6. 实验结果记录模板

### 6.1 微观指标结果

| 模型 | 1-step MSE | 5-step MSE | 10-step MSE | Fréchet ↓ | DTW ↓ |
|-----|-----------|-----------|------------|----------|-------|
| Baseline-A (RNN) | | | | | |
| Baseline-A (Transformer) | | | | | |
| Baseline-B (Data-only Diffusion) | | | | | |
| Model-C1 (Nav Cond) | | | | | |
| Model-C2 (+Macro Regularizer) | | | | | |

### 6.2 中观指标结果

| 模型 | 路径覆盖率 ↑ | 绕路率 ↓ | 行程时间 KL ↓ | 到达率 ↑ |
|-----|------------|---------|--------------|---------|
| Baseline-A | | | | |
| Baseline-B | | | | |
| Model-C1 | | | | |
| Model-C2 | | | | |

### 6.3 宏观指标结果

| 模型 | MSD α 误差 ↓ | 位移分布 KL ↓ | 活动半径 KL ↓ |
|-----|-------------|--------------|--------------|
| Ground Truth | $\alpha_\text{data}$ | — | — |
| Baseline-A | | | |
| Baseline-B | | | |
| Model-C1 | | | |
| Model-C2 | | | |

---

*最后更新：2025-12-09*

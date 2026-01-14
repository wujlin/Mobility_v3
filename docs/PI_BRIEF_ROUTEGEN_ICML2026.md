# ICML 2026 Route Generation｜新 PI 快速对齐（单一真相源）

> 目的：让新 PI 在 **15 分钟**内理解我们当前的 **数据口径、代码框架、实验路线、论文叙事**，避免被旧的（window-level / map-free / rupture）材料误导。  
> 更新时间：2026-01-14  
> 适用范围：ICML 2026 route generation（`essay_icml_cascadetraj/`）主线。

---

## 0) 一句话（当前主线）

我们研究 **segment-level 的 route generation**：给定 `(O, D, t0, context)` 生成整段路线；核心难点是 **corridor-level 多模态**。连续坐标端到端生成在长程上容易出现 **均值塌缩/漂移**。我们当前转向 **road-graph 上的结构化决策**：用 **少步数的 waypoint-level AR** 生成稀疏走廊承诺（commitment），再用 **A\*** 连接保证输出为合法图路径（后续再接 continuous execution/refinement）。

---

## 1) 关键转折：为什么从 window-level 改到 segment-level？

**结论**：window 采样会把任务降级成短距离轨迹延续（trajectory continuation），使“走廊选择”不存在，从而导致语义/拓扑相关实验全部失真。

**证据**（Detroit segments 上的统计审计）：
- `paired.detour_corr` 极低（prefix 与 full detour 几乎不相关），且 `lost_detour_frac_thr0p2` 显著：前缀窗口丢失了大比例绕行信息。
- 过滤后可用的 segment 数量足够（`keep` 数量约千级），支持 segment-level routegen。

这条结论已写入数据审计脚本产物（见 `docs/WORKSTATION_GUIDE.md` 约定的 `_sync/wsa/.../E_D0_*`、`E_W0_*`）。

---

## 2) 当前数据口径（你需要知道的“真相源”）

### 2.1 外置数据根目录（不进 git）

统一用 `$RAW_ROOT=/home/jinlin/data/geoexplicit_data`（工作站）：
- `worldtrace/detroit_core_v1/segments.parquet`（Detroit 片区子集）
- `worldtrace/columbus_core_v1/segments.parquet`（Columbus 片区子集）
- 全量数据：`worldtrace/OpenTrace_WorldTrace/`（3.6TB，当前未全量接入 pipeline）

### 2.2 RouteGen 训练用的 segment-level NPZ（固定长度）

我们把每条 trip segment 重采样到固定长度 `F=256`，并保证：
- `start_t` 为 **Unix epoch 秒**（时间特征 `temporal_mode=auto` 才会生效）
- `(O,D)` 来自 segment-level 起终点，而不是窗口前缀

典型产物（路径以工作站为准）：
- `$RAW_ROOT/experiments/icml2026_routegen/gt_segments/*_segments_route_F256_epoch_seed0.npz`

### 2.3 Road Graph 与 Graph Path 产物（map-aware track）

我们把 OSM 解析为 **road graph**，并把 GT route snap/map-match 成 **node sequence**：
- `road_graph_*.npz`：节点坐标、边、edge tier（major/minor/service）、网格 bbox/meta
- `paths_graph_*.npz`：每条 route 的 `start_node/dest_node/node_seq_pad/node_seq_len/start_t/route_city`

多城组合（Detroit+Columbus）当前使用：
- `$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/road_graph_combo.npz`
- `$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/paths_graph_combo.npz`

---

## 3) 代码框架（RouteGen 这一条线）

### 3.1 Road-graph 数据处理（T1/T2 的前置）

- GT 路线 → graph node sequence（map-matching / bridging）：  
  `src/data/road_graph/dump_graph_paths_from_routes_npz.py`
- 候选路径覆盖率诊断（K-shortest 覆盖不足的证据链）：  
  `src/data/road_graph/gate_candidate_paths_from_routes_npz.py`  
  `src/data/road_graph/diagnose_candidate_coverage.py`
- 分叉点/走廊聚类的语义信息量 Gate（AUC/MI）：  
  `src/data/road_graph/gate_semantic_informativeness_at_branch.py`（早期分叉点版本）  
  `src/data/road_graph/gate_semantic_informativeness_cluster.py`（聚类版本，当前推荐）

### 3.2 Decision：Waypoint AR（少步数、避免 600-step node AR 累积误差）

- GT graph path → 固定 K 的 waypoint 序列：  
  `src/data/road_graph/dump_waypoints_from_paths_graph_npz.py`
- Waypoint AR（bin 分类，控制输出空间）：  
  模型：`src/models/road_graph/ar_waypoint_bins.py`  
  训练：`src/training/train_graph_ar_waypoint_bins.py`

### 3.3 Execution：A* 连接（保证合法路径；后续可接 continuous executor）

- 采样 waypoint → A* 连接成完整 corridor path，并做 best-of-K 覆盖评估与可视化：  
  `src/training/sample_graph_ar_waypoints_astar.py`

> 备注：A* 是当前“执行层”的最小闭环，用于证明 **结构化决策能否恢复 corridor-level 覆盖**；continuous execution（diffusion/flow）可作为后续加分项，但不是当前收敛的前置条件。

---

## 4) 实验路线（当前有效的证据链）

### 4.1 诊断：端到端连续生成失败（Claim A）

- 在 segment-level route 上，L2 regression 会平均化；end-to-end diffusion 易 drift/off-road。  
  证据：segment-level 可视化对比（`E4s_*` 产物）。

### 4.2 为什么不用 “K-shortest + classify”（覆盖率不足）

- 候选覆盖 gate 显示：即使增大 K，GT 走廊覆盖率依然偏低（best Jaccard 低），无法支撑“候选集上做分类”的叙事。  
  证据：`G2_candidates_*` + `G2_diagnose_*`。

### 4.3 Go/No-Go：语义/时间是否能在 corridor 层提供可观测信号？

- 用 OD 分桶 + 走廊聚类得到二分类标签，计算 `AUC(time+tier)`。  
  当前结论：在少量可用 OD 组上 AUC>0.6（GO），但可用 OD 组数仍是瓶颈 → 需要更多城市 / 更多数据。

### 4.4 当前原型：Waypoint AR + A*

- 训练侧：waypoint-bin AR 在 1024 类上能学到非随机信号（val acc≈0.34）。
- 推理侧：A* 连接的成功率与 best-Jaccard 仍偏低，主要瓶颈是“bin→node 实例化缺少可达性约束”，会导致大量 A* 段不可达。

---

## 5) 论文叙事路线（ICML routegen，和代码/数据一致）

### 5.1 这篇论文讲什么

- 问题：route generation 的 corridor-level multi-modality 是结构化的；连续坐标端到端很难同时保证可行性与多样性。
- 洞见：走廊选择应当是 **结构化离散决策**（graph/waypoint commitment），而不是在坐标空间里“连续优化/平均”。
- 方法：CascadeTraj = Decision（waypoint commitment）+ Execution（合法连接/连续细化）。  
  当前实现：Decision=waypoint AR（少步数）；Execution=A*（确保合法）。  
  （可选扩展）Execution 可接 diffusion/flow 做 within-corridor 细化。

### 5.2 明确不讲什么（防止 PI/审稿人误解）

- 不讲 rupture/avoidance field（那是 paper-2，目录 `essay/`）。
- 不把 census/pop synthesis 当主线；城市语义仅在“可观测条件是否 informative”的 gate 中出现。
- 不用 window-level 的短片段作为 route generation 证据。

---

## 6) 你给新 PI 的“入口命令”（最短路径）

1) 读这份文档（本文件） + `docs/TASK_DEFINITION.md`（协议） + `docs/DATA_CONTRACT.md`（坐标/口径）。
2) 看 routegen essay：`essay_icml_cascadetraj/main.tex`。
3) 复现 T4 pipeline（waypoint AR + A*）：参考 `docs/WORKSTATION_GUIDE.md` 的 `wsA` 环境变量与 rsync 口径。

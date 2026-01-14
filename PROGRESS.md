Implementation Plan, Task List and Thought in Chinese

# 开发进度追踪（当前优先级：ICML 2026 Route Generation）

> [!IMPORTANT]
> **协议真相源**：`docs/TASK_DEFINITION.md` + `docs/DATA_CONTRACT.md`。  
> **新 PI 快速对齐**：先读 `docs/PI_BRIEF_ROUTEGEN_ICML2026.md`。  
> 本仓库同时保留两条写作线：  
> - ICML 2026 routegen：`essay_icml_cascadetraj/main.tex`（当前主线）  
> - Paper-2 rupture/avoidance：`essay/main.tex`（非当前 ICML 交付物）

---

## 0) 主线叙事（1 句话）

route generation 的 corridor-level 多模态在连续坐标空间里会诱发均值塌缩/漂移；我们转向 **road-graph 上的结构化决策**：用少步数的 **waypoint AR** 产生走廊承诺，再用 **A\*** 连接保证路径合法性（后续再接 continuous execution）。

---

## 1) 数据就绪情况（RouteGen）

> 工作站默认外置根目录：`$RAW_ROOT=/home/jinlin/data/geoexplicit_data`（见 `docs/DATA_STRUCTURE.md`）。

| 项目 | 当前状态 | 备注 |
|---|---|---|
| WorldTrace city 子集 | Detroit + Columbus | `scan_worldtrace_root` 显示仅这两城有 `segments.parquet` |
| Detroit segments | 2295 | `worldtrace/detroit_core_v1/segments.parquet` |
| Columbus segments | 5228 | `worldtrace/columbus_core_v1/segments.parquet` |
| Tier-road 准备 | Detroit ✅ / Columbus ❌ | Columbus 需要补 tier-road 预处理后才能做 time+tier gate |
| Segment-level routes NPZ | ✅ | 固定长度 `F=256`，`start_t` 为 epoch 秒 |
| Road graph（combo） | ✅ | `T3_combo_detroit_columbus_seed0/road_graph_combo.npz` |
| GT graph paths（combo） | ✅ | `T3_combo_detroit_columbus_seed0/paths_graph_combo.npz` |
| Waypoints graph NPZ | ✅ | `T4_wp_ar_astar_combo_seed0/T1_dump_waypoints/waypoints_graph.npz` |

---

## 2) 实验路线（RouteGen，当前有效）

### 2.1 诊断：continuous end-to-end 的失败模式（segment-level）

- 目标：证明 corridor-level 多模态下，L2 会平均化、diffusion 会 drift/off-road（segment-level 下更明显）。
- 产物：`E4s_*` 的可视化对比图（GT vs L2 vs E2E-Diff vs cascade 原型）。

### 2.2 Map-aware 诊断：K-shortest 候选覆盖不足（不作为最终解法）

- 目标：量化 `K-shortest` 作为候选集时对 GT 走廊的覆盖不足（best Jaccard 低）。
- 产物：`G2_candidates_*` + `G2_diagnose_*`（包含 GT-to-road 距离 p90 与候选 bestJ 分位数）。

### 2.3 Go/No-Go：语义/时间在 corridor 层是否 informative（Gate）

- 目标：验证 `(time + tier)` 对走廊簇标签的 AUC 是否显著高于 0.5（支持 “context-conditioned diversity” 的叙事）。
- 产物：`G3*_cluster_gate*/report.json`（重点看 used_groups 与 AUC 分布）。

### 2.4 当前原型：Waypoint AR + A*（T4）

- 训练：`train_graph_ar_waypoint_bins.py`（val acc≈0.34，说明不是随机猜 bin）
- 评估：`sample_graph_ar_waypoints_astar.py`（当前瓶颈：bin→node 实例化缺少可达性约束，导致 success_rate 偏低）

---

## 3) 下一步（给 PI 的决策点）

1) Columbus 补齐 tier-road（解锁多城 time+tier gate 与 AR 训练一致口径）。
2) 改善 waypoint 采样的“可达性”（bin→node 需要连通性/可达性约束），优先把 `success_rate` 从 ~0.1 拉到可用区间。
3) 补齐对照：在同一评估脚本里输出 `K-shortest` baseline 的 bestJ（避免只看单模型）。

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

route generation 的 corridor-level 多模态在连续坐标空间里会诱发均值塌缩/漂移；我们将主线切换到 **map-aware 的 segment 表达**：用 **feature-based segment encoding（不依赖 segment ID）** + **latent compression** 把可变长路线压缩到固定长度 latent，再在 latent 空间用 **flow matching / diffusion** 建模可采样的分布，最后用 **连通性约束的解码（adjacency mask + beam）** 输出合法的 graph 路线（CASD：Corridor-Aware Segment Diffusion）。

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
| CASD segment graph & route segments | ✅（本地验证已跑通） | 入口脚本 `run_casd_prep.sh`，产物见 `CASD0_segdata_combo_seed0/`（新目录，不覆盖旧实验） |

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

### 2.4 当前原型：Waypoint AR + A*（T4，作为 baseline/对照）

- 训练：`train_graph_ar_waypoint_bins.py`（val acc≈0.34，说明不是随机猜 bin）
- 评估：`sample_graph_ar_waypoints_astar.py`（已修复跨城混选导致的不可达：`success_rate≈1.0`；当前瓶颈转移为 **bin 粒度过粗导致走廊混淆**，oracle 上界仅 `bestJ≈0.276`）
  - 重要口径：这里的 `bestJ` 默认是“单条 GT vs K 次采样”的 single-trajectory match；corridor diversity 需要基于同 OD/同 OD-bin 的多实例 GT（见 `src/data/road_graph/od_group_stats_paths_graph_npz.py` 与 `docs/PI_BRIEF_ROUTEGEN_ICML2026.md`）。

---

### 2.5 新主线：CASD（Corridor-Aware Segment Diffusion）

> 目标：把“走廊多模态”从 **候选枚举/长 AR** 转成 **latent 分布建模**，并保持 **图连通性** 与 **跨城泛化**（feature-only，不用 ID）。

**CASD0｜数据准备（已实现脚本）**
- 输入：`T3_combo_detroit_columbus_seed0/{road_graph_combo.npz, paths_graph_combo.npz}`
- 处理：
  - `build_segment_graph_from_road_graph_npz.py`：把 raster road graph 的 directed edges 折叠成 “segment graph”（degree-2 chain collapse）
  - `dump_segment_sequences_from_paths_graph_npz.py`：把每条 GT node path 映射成 segment-id 序列
- 入口：`run_casd_prep.sh`
- 产物：`CASD0_segdata_combo_seed0/S1_segment_graph/segment_graph.npz` + `S2_segment_routes/segments_graph_routes.npz`（均带 `report.json`）

**CASD1｜数据一致性 Gate（必须先过）**
- 问题：segment 序列展开后是否能复原原始 node path（或 edge set）？
- 通过标准：`missing_edge_frac=0`，且复原路径的 edge-Jaccard（vs 原 node_seq）接近 1（容忍极少量桥接/边缺失）

**CASD2｜压缩可行性 Gate（Autoencoder，先不引入 POI/Sat）**
- 问题：固定长度 latent（例如 L=8, d=256）能否保留 route 的走廊结构？
- 方法：feature-based SegmentEncoder + Perceiver 压缩 + 受 adjacency 约束的解码器（避免对 100 万 segment 做全量 softmax）
- 通过标准：重构 best-Jaccard 明显高于 `K-shortest` baseline；可视化能看到“走廊级别的合理替代”而非随机乱走

**CASD3｜Latent Flow / Diffusion（CASD 核心）**
- 问题：在 latent 空间能否采样出多样但可解码的路线分布？
- 方法：复用 `src/models/flow/rectified_flow_model.py` 的思想（flow matching），但把输入换成 `(B,L,d)` latent；条件 `c=(O,D,time,city)`（corridor_type/CFG 先占位）
- 通过标准：best-Jaccard / diversity 指标优于 AE-only；可视化显示“多走廊覆盖”而不是发散

**CASD4｜Urban semantics 注入（POI / Wayback，作为 data contribution）**
- 问题：语义/视觉是否在走廊选择上提供可观测增益（而非引入偏置）？
- 方法：Stage1 cross-attention；用 intervention audit（none/zeros/shuffle/flip）验证确实使用了空间结构

### 2.6 架构 Review（当前 CASD 需要注意的两个硬约束）

1) **segment 词表极大（约百万级）**  
Stage4 若用全量 softmax（`Linear(d)->n_segments`）会不可训练；必须用 **候选受限的 pointer/scoring**（例如只在 `seg_succ` 邻接集合上打分）或检索式解码。

2) **segment 序列长度目前 p50≈229（combo 数据）**  
若 Stage4 是逐步 AR 解码，仍会有长序列累积误差风险；CASD2/3 应优先验证“latent→走廊结构”的有效性，必要时引入更粗粒度 segment 定义（例如基于更高层 OSM edge，而非 raster cell）。

## 3) 下一步（给 PI 的决策点）

1) 先跑 CASD0→CASD1（数据一致性 Gate），确认 segment 表达正确无歧义（这是新主线的地基）。
2) CASD2（AE Gate）决定 “latent 压缩是否可行”；若不可行，优先调整 segment 粒度/定义，而不是盲目堆模型。
3) 旧主线 T4（waypoint-bin AR + A*）保留为对照与 fallback：它提供一个“map-aware 结构化决策”的 baseline，用于衡量 CASD 的增益是否真实。

# State-of-the-Art (SOTA) in Urban Trajectory Generation (2023-2024)

本文档梳理当前轨迹生成领域的两大主流技术路线：**Map-based Diffusion (路网图扩散)** 与 **Hierarchical Planning (分层生成)**。这两个方向主要解决 Grid-based / End-to-End 方法中存在的 "Destination Gravity"（过早收敛）和 "Topological Collapse"（绕路困难）问题。

---

## 1. Map-based / Graph-based Diffusion (路网流派)

该流派的核心思想是将轨迹生成从**欧氏空间 (Euclidean Grid)** 转移到 **路网图空间 (Road Graph)**。模型不再直接预测 $(x, y)$ 坐标，而是预测图上的 **节点序列 (Node Sequence)** 或 **边转换 (Edge Transition)**。

### 1.1 核心优势

- **拓扑合法性 (Topological Validity)**：天然保证轨迹在路网上，避免“穿墙/飞越街区”。
- **长距离导航 (Long-horizon Navigation)**：借助图结构与搜索先验，更容易学习“先上高速、再下匝道”这类宏观决策。
- **解决 Destination Gravity**：路网结构强制模型沿可达路径前进，不会被终点“直线吸”过去。

### 1.2 代表性工作（按你们之前调研口径记录）

> 说明：此处只保留我们当前讨论中已经出现过的代表作名称与会议归属；若需要严格引用（作者/年份/BibTeX），请在最终写论文前补全。

- **DiffTraj (SIGSPATIAL 2023)**
- **G-Diff (KDD 2024)**

### 1.3 对我们的启示

- 若要“根治拓扑坍缩”，引入 road graph 是强方案，但工程门槛高（map-matching、图构建、对齐）。

---

## 2. Hierarchical / Coarse-to-Fine Planning (分层流派)

该流派的核心思想是 **“先定骨架（宏观决策），再填细节（微观执行）”**。将生成过程分解为两个阶段：
先生成少量关键的 **Waypoints (途径点)**，再在 Waypoints 之间生成细粒度轨迹点。

### 2.1 核心优势

- **打破终点引力**：Waypoints 充当“中间锚点”，避免被终点势场直接吸走。
- **无需路网 (Map-free)**：可在 grid/连续空间工作，不需要复杂路网预处理。
- **更贴合 trip-level 本质**：把“路线选择（多模态决策）”与“局部执行（连续控制）”解耦。

### 2.2 代表性工作（按你们之前调研口径记录）

- **Hierarchical Diffusion for Mobility (NeurIPS 2024)**

### 2.3 对我们的启示

- 在不引入强路网数据的前提下，解决 Destination Gravity 的最佳主线通常是 **Hierarchical / Coarse-to-Fine**。
- 我们的 Residual Diffusion 可被视作“退化的分层”：Prior 类似粗路径；但如果粗路径本身缺少 detour 拓扑，那么 residual 很难凭空创造绕路。

---

## 3. 总结与建议

| 特性 | Grid-based (Ours) | Map-based (SOTA) | Hierarchical (SOTA) |
| :--- | :--- | :--- | :--- |
| **拓扑合法性** | 弱 (可能穿墙) | 强 (路网约束) | 中 (取决于 waypoint) |
| **绕路能力** | 弱 (Destination Gravity) | 强 (图结构引导) | 强 (中间锚点引导) |
| **数据要求** | 低 (仅 GPS) | 高 (GPS + 路网) | 低 (仅 GPS) |
| **工程复杂度** | 低 | 高 | 中 |

**结论**：在当前的资源与时间约束下，建议优先走 **Hierarchical Planning** 路线；只有当分层路线被证伪，才考虑引入 road graph / map constraints。


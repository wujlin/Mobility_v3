# State-of-the-Art (SOTA) in Urban Trajectory Generation (2023-2024)

本文档梳理当前轨迹生成领域的两大主流技术路线：**Map-based Diffusion (路网图扩散)** 与 **Hierarchical Planning (分层生成)**。这两个方向主要解决 Grid-based / End-to-End 方法中存在的 "Destination Gravity"（过早收敛）和 "Topological Collapse"（绕路困难）问题。

---

## 1. Map-based / Graph-based Diffusion (路网流派)

该流派的核心思想是将轨迹生成从**欧氏空间 (Euclidean Grid)** 转移到 **路网图空间 (Road Graph)**。模型不再预测 $(x, y)$ 坐标，而是预测图上的 **节点序列 (Node Sequence)** 或 **边转换 (Edge Transition)**。

### 1.1 核心优势
*   **拓扑合法性 (Topological Validity)**：天然保证轨迹在路网上，绝对不会“穿墙”或“飞越街区”。
*   **长距离导航 (Long-horizon Navigation)**：借助图搜索算法（如 Dijkstra/A*）的先验知识，模型更容易学习到“先上高速、再下匝道”这类宏观决策。
*   **解决 Destination Gravity**：路网结构强制模型遵循物理连接，即使目标很远，模型也必须沿着路网一步步走，不能直线“吸”过去。

### 1.2 代表性工作

#### **DiffTraj (SIGSPATIAL 2023)**
*   **方法**：将轨迹视为路网图上的随机游走。使用 Graph Neural Network (GNN) 编码路网结构，使用 Diffusion Model 生成路段（Edge）的属性（如通行时间）。
*   **特点**：强约束，生成的每一条轨迹都是路网合法的。
*   **局限**：对路网数据质量要求极高（Road Network Completeness），且推理速度受限于图规模。

#### **G-Diff (KDD 2024)**
*   **方法**：结合了 Graph Diffusion 和 Spatial-Temporal Graph Neural Networks (ST-GNN)。在生成过程中显式地建模了交通流的动态变化。
*   **特点**：不仅生成位置，还能生成速度和流量，适合交通仿真。

### 1.3 对我们的启示
*   如果我们想根治 "Topological Collapse"，引入 Road Graph 是必经之路。
*   **工程门槛**：需要实现高精度的 Map Matching（将 GPS 点投影到路网）和复杂的图数据预处理。

---

## 2. Hierarchical / Coarse-to-Fine Planning (分层流派)

该流派的核心思想是**“先画骨架，再填肉”**。将生成过程分解为两个阶段：先生成几个关键的 **Waypoints (途径点)**，再在 Waypoints 之间生成细粒度的轨迹。

### 2.1 核心优势
*   **打破终点引力**：Waypoints 充当了“中间锚点” (Intermediate Anchors)。模型在第一阶段只需要关注“大概走哪条路”（比如选哪个桥过河），从而避免了被终点直接吸走。
*   **无需路网 (Map-free)**：依然可以在 Grid 或连续空间上工作，不需要复杂的路网预处理。
*   **计算高效**：长序列生成被拆解为多个短序列生成，降低了累积误差。

### 2.2 代表性工作

#### **Hierarchical Diffusion for Mobility (NeurIPS 2024)**
*   **方法**：
    1.  **Macro-Stage**: 给定 OD，生成 $K$ 个 Waypoints（例如时间分位点 $t=0.25, 0.5, 0.75$ 的位置）。
    2.  **Micro-Stage**: 给定相邻的 Waypoints，并行生成中间的详细轨迹点。
*   **特点**：显著提升了长距离轨迹的绕路能力（Detour Capability）。
*   **局限**：如果 Macro-Stage 生成的 Waypoint 在不可行区域（如湖里），Micro-Stage 很难救回来。

#### **Planning-guided Diffusion**
*   **方法**：利用 A* 或 RRT 等传统规划算法生成一条粗糙的“参考路径” (Reference Path)，然后用 Diffusion Model 对其进行平滑和细化。
*   **特点**：结合了传统规划的逻辑性和生成模型的多样性。

### 2.3 对我们的启示
*   这是在不引入路网数据的前提下，解决 Destination Gravity 的最佳方案。
*   **低成本改进**：我们当前的 Residual Diffusion 其实可以看作是一种退化的分层模型（Prior = 粗糙路径）。如果我们将 Prior 替换为基于 Waypoint 的生成，或者让 Residual 学习“相对于 Waypoint 的偏移”而不是“相对于 Prior 的偏移”，效果可能会大增。

---

## 3. 总结与建议

| 特性 | Grid-based (Ours) | Map-based (SOTA) | Hierarchical (SOTA) |
| :--- | :--- | :--- | :--- |
| **拓扑合法性** | 弱 (可能穿墙) | 强 (路网约束) | 中 (取决于 Waypoint) |
| **绕路能力** | 弱 (Destination Gravity) | 强 (图结构引导) | 强 (中间锚点引导) |
| **数据要求** | 低 (仅 GPS) | 高 (GPS + 完整路网) | 低 (仅 GPS) |
| **工程复杂度** | 低 | 高 | 中 |

**结论**：
我们目前的 Grid-based 方法在物理微观真实性（Physics Realism）上有优势，但在宏观拓扑决策上存在局限。未来的工作应优先考虑 **Hierarchical Planning** 路线，因为它能以较低的工程成本显著改善绕路问题，且不需要依赖外部路网数据，更符合 Data-driven 的初衷。


# SOTA 轨迹生成技术路线梳理与定位 (State-of-the-Art Trajectory Generation Roadmap)

> **核心背景**：针对 Phase B 中发现的 "Destination Gravity"（直冲终点）与 "Mode Collapse"（回归均值/走直线）问题，本文档梳理了学术界（2023-2024）的主流解决方案，并确立我们 **"Hierarchical Waypoint"** 方案的学术定位。

---

## 1. 核心痛点：为什么 End-to-End Diffusion 会失败？

在长距离（Trip-level）轨迹生成任务中，单纯的 `Diffusion(O, D) -> Trajectory` 面临 **"Topological Uncertainty"（拓扑不确定性）**：

1.  **多模态灾难**：从 A 到 B 有多条宏观路径（上路/下路）。
2.  **均值回归（Regression to Mean）**：当模型无法用足够的容量区分多个模态时，它会学习概率分布的**均值**。
3.  **结果**：$\text{Mean}(\text{Route A}, \text{Route B}) = \text{Straight Line (Infeasible)}$。生成的轨迹穿墙而过，物理不可行。

---

## 2. 学术界主流 SOTA 解决方案 (The Three Schools)

为了解决上述问题，SOTA 模型普遍采用了 **"分而治之" (Divide and Conquer)** 的策略，将 **"决策 (Decision)"** 与 **"执行 (Execution)"** 解耦。

### 流派一：Coarse-to-Fine (网格/区域离散化)
**代表作**：*CityFlow, DiffTraj-Grid*

*   **核心思想**：将连续的城市空间离散化为 **Grid (网格)** 或 **Region (区域)**。
*   **流程**：
    1.  **Macro (Decision)**：预测离散的区域序列（Region Sequence）。
        *   *优势*：这是一个分类问题（Classification），天然支持多模态，不存在“均值”问题。
    2.  **Micro (Execution)**：在确定的网格序列内生成连续坐标。
*   **局限性**：精度受限于网格大小，容易产生锯齿状（Zigzag）路径，且丢失路口细节。

### 流派二：Anchor-based / Heatmap (热力图与锚点)
**代表作**：*Y-Net, TNT (Target-driven Trajectory Prediction)*

*   **核心思想**：不直接预测坐标，而是预测 **概率密度函数 (PDF)**。
*   **流程**：
    1.  **Heatmap Prediction**：预测未来 $T$ 时刻或关键位置的 2D 热力图。
    2.  **Peak Sampling**：从热力图的**波峰 (Peaks)** 采样出关键点（Waypoints/Goals）。
        *   *优势*：显式地捕获了多模态（双峰分布），避开了低概率的“山谷”（直线路径）。
    3.  **Trajectory Completion**：连接起点、关键点与终点。
*   **局限性**：预测高分辨率热力图计算量大，且后处理（从图取点）较繁琐。

### 流派三：Temporal Hierarchies / Leapfrog (时间金字塔)
**代表作**：*Leapfrog Diffusion, Midpoint Recursion*

*   **核心思想**：**递归生成 (Recursion)**。
*   **流程**：
    1.  **Step 1**：给定 $P_0, P_T$，预测中间点 $P_{T/2}$。
    2.  **Step 2**：给定 $P_0, P_{T/2}$，预测 $P_{T/4}$... 以此类推。
*   **优势**：将长程依赖拆解为多个短程依赖，误差不累积。
*   **我们的借鉴**：这正是我们引入 Waypoint 的理论基础——先定中间，再定两头。

---

## 3. 本方案：Data-driven Hierarchical Waypoint

本方案结合了 **流派二 (Anchor)** 和 **流派三 (Hierarchy)** 的优势，提出 **Map-free Hierarchical Framework**。

### 3.1 架构定义

| 层级 | 任务性质 | 输入 | 输出 | 对应模型 | 物理含义 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Level 1** | **Trip-level (Decision)** | $O, D, t_0$ | **Waypoint $W$** | CVAE / MLP | **Navigator (领航员)**<br>负责拓扑决策，锁定“走哪条路”。处理多模态。 |
| **Level 2** | **Segment-level (Execution)** | $O \to W, W \to D$ | **Trajectory $\tau$** | Physics Diffusion | **Driver (车手)**<br>负责物理执行，锁定“怎么开顺”。处理局部平滑。 |

### 3.2 为什么这依然是 Map-free？

这是本方案的核心创新点（Paper Contribution）：

*   **Explicit Map (传统方法)**：推理时需要加载 Road Graph 文件，依赖 A* 搜索。泛化差，依赖地图数据质量。
*   **Implicit Map Learning (本方案)**：
    *   模型不依赖地图输入。
    *   通过观察海量轨迹，**自动涌现 (Emerge)** 出对关键节点的认知。
    *   Waypoint 是**从数据中蒸馏出来的隐式地图知识**。
    *   **结论**：这是 **Strict Map-free**，且具备更强的泛化性和鲁棒性。

---

## 4. 验证计划 (Action Plan)

为了避免再走 “看起来合理但实际上是同义反复” 的弯路，分层路线必须先通过两个 **Go/No-Go Gate**（CPU-only，不烧卡），再做 Oracle 执行能力诊断。

> **硬约束协议（必读）**：`docs/archive/legacy_shenzhen/HIERARCHICAL_VALIDATION_PROTOCOL.md`

### 4.0 Waypoint Gate（Go/No-Go；先跑这个）

> 目的：验证“粗层 waypoint/skeleton”这套表征是否 **(a) 可行** 且 **(b) 有信息量/可学**。  
> 如果连这一步都过不了，就不要进入 hierarchical（否则宏观层学不到因果信号，微观层也无法被正确引导）。

**A) 物理合法性检验（Validity Check）：Skeleton 碰撞率**
- 从 GT 提取少量 waypoints（如 `max_dev / max_turn`；并用 `time/random` 做基线）。
- 生成 skeleton：`start → waypoint(s) → end` 直接连线（可选 spline 仅作诊断）。
- 在弱图（map-free proxy）上做碰撞检测：默认用 `nav_field.npz` 的 `count` 构建 drivable mask（`count>=thr`），并允许小范围 `close/dilate` 以抵消离散化/稀疏采样误差。
- **硬阈值**：若 `collision_rate_any > 10%`，直接 **否决该 waypoint 定义/该 coarse 表示**（不用再看 MSE/ADE）。

**B) 信息量检验（Learnability Check）：几何特征相关性**
- 计算 waypoint 与几何特征点（路口/瓶颈的 proxy）的相关性：例如 waypoint 到 corner/junction proxy 的距离分布；并与 `time/random` 基线对比。
- 判据：若与基线无差异（或相关性≈0），说明 waypoint 更像“均匀时间点”，宏观层很难学到可泛化的导航策略。

脚本：`src/evaluation/waypoint_gate.py`（依赖 SciPy，CPU-only）

```bash
python -m src.evaluation.waypoint_gate \
  --samples_npz data/experiments/prior_geo_density_test/samples.npz \
  --nav_file data/processed_dt30/nav_field.npz \
  --waypoint_mode max_turn --num_waypoints 1 \
  --skeleton linear+pchip \
  --count_thr 1 --close 0 --dilate 0 \
  --out_json data/experiments/waypoint_gate/prior_density10k_maxturn_k1_thr1.json
```

> 现状快照（基于 `prior_geo_density_test/samples.npz`, N=10k）：  
> - `count_thr=1, close=0, dilate=0`：linear ≈ 6–8%（<10%），pchip ≈ 12–16%（>10%，对 mask 很敏感）  
> - `count_thr=1, close=1, dilate=1`：linear/pchip 均显著下降（<10%）  
> - Learnability（corner 距离/相关性）当前接近基线（≈0）→ **仍需更强的“几何特征 proxy”或更合适的宏观变量定义（不一定是单点 waypoint）**。

### 4.1 Oracle Execution（诊断 micro 是否具备“执行 detour”的能力）

注意：**仅做 “GT waypoint → 插值回 GT” 是同义反复**；真正有效的 Oracle 是把 waypoint 当作条件喂给现有生成器跑推理。

1. **Oracle Extraction (GT)**：从测试集 GT 轨迹中提取 $W^*$（如 `max_dev`）。
2. **Segmented Inference**：使用现有 Phase B Diffusion/Physics 模型，分别推理 $O \to W^*$ 与 $W^* \to D$（无需重训）。
3. **Metrics Check**：看 `JSD_TurnAngle` 是否下降（更接近 GT），并结合 DCV/可视化。

> 当前结论：OracleWP 已被证伪（详见 `docs/archive/phase_b/PHASE_B_CFG_VISUALIZATION.md#8.1.1`），说明现有生成器的 support 基本没有“低频、平滑 detour”模态。  
> 因此：hierarchical 若要成立，**不能只补宏观层**，还需要重训/重构 segment-level executor（更短 horizon / 更强条件 / 更结构化 latent）。

### 4.2 进入训练前的决策树（避免再浪费一周）

- 若 **4.0(A) Validity fail**：分层 coarse 表征不可用 → 直接考虑 weak map / road graph（工程量级更大，但方向更对）。
- 若 **4.0 pass 但 4.1 fail**：宏观层可行但微观执行不具备 → 优先修 executor（不要先训 waypoint predictor）。
- 若 **4.0 pass 且 4.1 pass**：才进入 Learned Macro（训练 waypoint predictor / route code），并用 micro executor 组合评估。

---

## 5. 术语对照表 (Terminology)

*   **Trip-level**: 宏观行程，涉及路径选择（Routing）。
*   **Segment-level**: 微观路段，涉及车辆控制（Control/Physics）。
*   **Topological Commitment**: 拓扑承诺。一旦选定某条路（如上高速），后续轨迹被锁定，无法轻易切换。
*   **Implicit Map Learning**: 隐式地图学习。从轨迹数据中学习路网结构，而非直接读取地图文件。
*   **Mode Collapse**: 模态坍缩。模型丢失了多模态特性，退化为单一的（通常是错误的）均值输出。


---

## 6. 2025 年底更新：SOTA 新趋势与代表作（补全 2025-01 ～ 2025-12）

> 本节在你原有的 2023-2024 技术路线（流派一/二/三 + Hierarchical Waypoint）基础上，补充截至 **2025-12-24** 能检索到的代表性工作。
> 重点关注：**如何更好地处理 Trip-level 的拓扑不确定性（多模态）、更强的可控性/泛化性、以及更高的生成效率**。

### 6.1 Backbone 升级：UNet → Transformer（提升容量与细节保真）

- **Traj-Transformer (arXiv:2510.06291, 2025)**  
  观察：传统 UNet/卷积噪声预测器在 GPS 轨迹扩散生成中容易出现偏移、丢失街道级细节（容量不足）。  
  做法：用 **Transformer** 同时做条件嵌入与噪声预测，并比较两种 GPS 点嵌入策略（location embedding / lon-lat embedding）。  
  启示：该方法揭示了 Transformer 架构在条件嵌入与噪声预测中的潜力。若当前扩散模型采用卷积架构（如 UNet），迁移至 Transformer 是提升模型容量与细节保真度的可行路径。

### 6.2 Coarse-to-Fine 2.0：离散“路段/区域” + 连续“GPS细粒度”（结构化分解更明确）

- **Cardiff (arXiv:2507.13366, 2025)**  
  典型的“结构化分解”：先在 **离散 road segment-level** 上做 latent diffusion（论文明确提到 diffusion transformer），再做 **连续 GPS-level** 的 conditional denoising，并加入 noise augmentation，且可调节 privacy-utility。  
  启示：该工作与本文档提出的“Decision / Execution 解耦”理念高度一致，并将决策层进一步具象化为 *road-segment 序列*，值得在架构设计上作为重要参考。

- **GeoGen (arXiv:2510.07735, 2025；AAAI 2026 接收信息见公开渠道)**  
  面向 **LBSN/不规则离散空间** 的两阶段 coarse-to-fine：先把离散序列重建到连续规则 latent（diffusion），再用 seq2seq 生成细粒度序列（Transformer）。  
  启示：对于涉及“网格/区域离散化”的技术路线，GeoGen 提出的“先连续潜空间、再细粒度离散”的建模方式提供了另一种有效的对标方案。

### 6.3 Hybrid：AR/序列决策 + Diffusion 执行（“先给路线骨架，再补坐标细节”）

- **Traveller (Information Fusion, 2025/2026 期刊排期；doi:10.1016/j.inffus.2025.103766)**  
  提出“Autoregressive Diffusion Model”：**AR-TempPlan** 负责时间/出行模式规划，**TravCond-Diff** 负责空间轨迹生成；并用 home location + mask location sequence 表达 travel pattern。  
  启示：这是“Trip-level 决策 → Segment-level 执行”框架的典型实现，其将宏观决策明确建模为“出行模式/时间规划”，为分层生成提供了具体范例。

- **Seed (The Web Conference / WWW 2025)**  
  目标是“桥接 Sequence 与 Diffusion”：把序列模型的规律性 + 扩散模型的多样性结合，用于 road trajectory generation。  
  启示：针对“纯 diffusion 难以捕捉长期规则”的问题，该研究证明了结合自回归（AR）生成骨架序列与扩散模型细化的有效性。

### 6.4 噪声先验与约束：让 diffusion 更“懂城市”（个体 + 群体 + 物理）

- **CoDiffMob (arXiv:2412.05000, 2024/2025 修订)**  
  核心点是“Noise matters”：指出直接用 i.i.d. 噪声忽略了城市移动中的时空相关与群体交互，提出 collaborative noise priors 来指导生成，报告 >32% 改进。  
  启示：该方法直接对扩散过程中的噪声分布进行建模，而非仅调整网络结构，为优化生成过程提供了新的切入点，尤其适用于提升时空相关性。

- **Dynamic Population Distribution Aware TG (arXiv:2511.01929, 2025)**  
  把“动态人口分布”作为条件/约束显式注入 diffusion 与 denoising；并构建空间图增强空间相关。  
  启示：当任务目标包含“宏观 OD/热区分布一致性”时，引入此类宏观动态分布作为显式约束或条件是提升模型表现的有效手段。

- **GCDM / Geo-lucid Conditional Diffusion (SIGSPATIAL 2025)**  
  讨论“physical fidelity（几何 + 动力学）”，并提出把 road map attributes 通过 spatially hierarchical generation 与 map-informed latent variables 融入扩散生成。  
  启示：即便在 map-free 设定下，该工作也提示了一个重要方向：**利用层级潜变量（Hierarchical Latent Variables）来承载“隐式地图知识”**，这与 Waypoint 方案的学术定位相符。

### 6.5 更快更大尺度：Flow Matching / ODE 生成（diffusion 的“加速路线”）

- **TrajFlow (OpenReview，ICLR 2026 投稿版本在 2025-12 有公开记录)**  
  从“国家尺度/多尺度”角度批评扩散模型采样步数多、难扩展，并提出 flow matching 框架。  
  启示：在涉及更大空间尺度或长序列生成的场景中，Flow Matching 等 Score/ODE 家族模型因其采样效率优势，是值得关注的潜在替代方案。

### 6.6 轨迹 Foundation Model 化：统一多任务/跨城市迁移（给 Trip-level Decision 层提供更强表征）

- **TrajFM (arXiv:2408.15251, 2024)**  
  明确把“region transfer + task transfer”当作目标，用 STRFormer（多模态：空间/时间/POI）+ masking & recovery 统一多任务生成。  
  启示：对于 Level-1 的 Trip-level 决策模块，采用 Foundation Model 式的预训练（如 Masking & Recovery）可能比从零训练 CVAE/MLP 具备更强的跨域泛化能力。

- **TrajGPT (KDD 2024 / arXiv:2411.04381)**  
  把“受控合成轨迹生成”类比为 LLM 的 text infilling：在 Transformer 架构里联合建模空间与时间，并强调 spatiotemporal consistency。  
  启示：此类“填空式生成”与“给定 O、D 及部分中间约束点”的任务形式同构，为基于 Waypoint 的生成方案提供了建模思路。

- **SIGSPATIAL Vision 2025：Toward Foundation Models for Mobility Enriched GEOs**  
  从更宏观角度讨论 GeoAI 缺少可迁移表征，以及 mobility 数据在 GEO 表征学习中的角色。  
  启示：“Implicit Map Learning”可被阐释为“Mobility-enriched GEO Representation Learning”的具体实践，提升了方法的理论高度。

### 6.7 Map-conditioned / Zero-shot / Language-conditioned：更强的“可控生成接口”

- **Map2Traj (IJCAI 2025)**  
  仅用 street map 作为输入就能对“未观测区域”做 zero-shot trajectory generation（面向无线网络优化应用）。  
  启示：该工作证明了“地图作为条件”能显著提升跨区域泛化能力，可作为 Map-free 研究的重要对照基准（Upper Bound）。

- **GTG (AAAI 2025)**  
  通过学习跨城市 invariant mobility patterns，并结合 shortest path search 来实现泛化生成。  
  启示：表明“跨城泛化”的关键在于解耦“可迁移规律（Invariant）”与“城市特定拓扑”，Waypoint 可被解释为一种“可迁移的拓扑决策变量”。

- **LangTraj (ICCV 2025)**  
  把自然语言作为条件做交通/轨迹模拟，并提出 closed-loop training 来减少闭环误差累积。  
  启示：尽管任务形式不同，但“语言/意图 → 轨迹”的可控生成接口（Intent-conditioned Generation）为 Profile-conditioned 生成提供了参考。

---

## 7. 总结：Hierarchical Waypoint 的“2025 对齐结论”

综上所述，2025 年轨迹生成领域的主流趋势集中于强化**“拓扑/出行模式的显式决策变量”（如 Waypoint、Segment、Pattern 或 Map Condition）**，并结合更强大的生成器（如 Transformer-Diffusion、Flow Matching）以实现高保真的细节执行。启示：这一趋势与本文提出的 **“Decision / Execution 解耦”** 及 **“Hierarchical Waypoint”** 方案高度契合，确立了该方案在当前技术前沿中的合理定位。

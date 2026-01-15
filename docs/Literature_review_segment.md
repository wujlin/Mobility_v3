下面按你列的 **4 个核心不确定性** 来做一份 **2024–2026**（含线上先发/预印本）导向的文献梳理。整体主线我会把“人类路线选择/道路轨迹生成”拆成四层：

1. **表征层（features / road representation）**：路段怎么编码、上下文怎么融入
2. **动作层（候选集 / action space）**：从多大空间里“选下一步/选下一段”
3. **抽象层（Decision Point / hierarchy）**：什么算“决策点/粗粒度”
4. **生成层（AR vs non-AR / diffusion / search）**：用什么生成范式建模决策序列

---

## 不确定点 1：特征设计是否足够？

你现在的特征（相对位置、道路等级、方向角…）属于**局部、几何与语义轻量特征**。2024–2026 的 SOTA 趋势是：

* 在“局部几何/语义”之外，**显式加入网络拓扑与全局结构特征**（比如 betweenness、integration 等），以及 **时间/出行条件（OD、出发时间、轨迹统计量）**；
* 同时用模型结构把“粗到细”的信息分层编码（减少你手工特征要背的锅）。

### 1.1 Cardiff (arXiv:2507.13366) 的 road segment representation 用了哪些特征？

Cardiff 把 **路段序列当成 token 序列**来建模（segment-level → latent diffusion → 再条件生成 GPS-level）。它对“路段”概念的定义里，明确区分了：

* **路段属性**：长度、类型、速度等
* **几何特征**：路段的 GPS 坐标序列（polyline）

在具体的 segment-level 编码里，它采用“token + 坐标”的做法：

* 每个路段当作一个 token
* **把路段中心点坐标嵌入到 token embedding 里**（用来注入空间信息）
* 解码时提到使用 **beam search** 来恢复 segment 序列

此外，Cardiff 的“条件”不仅是路网特征，还包括**轨迹级条件**：

* 出发时间、起点、终点
* 轨迹统计量（例如轨迹长度、持续时间）

它也明确在数据/路网侧使用了来自 OSM 的信息（路段类型、长度、是否单行、节点边的经纬度等）作为道路网络特征来源。

> 你可以把 Cardiff 的启发理解为：
> **“少做复杂手工特征，但一定要给模型足够的空间锚点（center coord / geometry）+ 轨迹条件（OD+time）。”**

### 1.2 GTG (AAAI 2025) 用了哪些特征来学 “city-invariant mobility patterns”？

GTG 的关键点是 **City-invariant road representation**：

* 它强调用 **Space Syntax** 提取城市无关的拓扑结构特征（这就是 “invariant mobility patterns” 的核心抓手之一）
* Space Syntax 特征包括：**Total Depth、Integration、Connectivity、Choice/Betweenness**

在此基础上，GTG 还用了一组“基础路段特征”：

* 路段长度、类型、方向等

以及与时间相关的处理：

* 把时间切片的 index 作为输入来刻画随时间变化的 travel cost

更进一步，它在建模路段之间关系时，用 SAGAT（Spatial Awareness GAT）显式编码**路段对的关系特征**：

* 路段对的 betweenness、转角（turning angle）、中心点距离/旅行距离等

**这对你的启发很直接**：
你现在的“方向角/道路等级/相对位置”类似 GTG 的“基础特征”，但 GTG 证明（至少在它的问题设定中）**拓扑结构特征（Space Syntax）是重要增益**。

### 1.3 有没有“什么特征最重要”的消融实验？

**GTG 有比较直接的消融证据**：

* ablation 中包括去掉 Space Syntax（w/o SS）等设置；作者指出去掉 cost prediction 模块影响最大，同时 **Space Syntax 的移除也显示出显著贡献**

**Cardiff 的消融更偏“表示学习模块”的消融**：
例如它有一个变体会去掉 latent 压缩和外部坐标嵌入，用来评估 compressed encoding 和坐标嵌入的重要性。

---

### 1.4 你现在特征是否“足够”？用文献主线给一个可操作结论

对照 GTG/Cardiff，你现在特征集“可能够用”去拟合一些局部偏好，但**容易缺两类关键因素**：

1. **网络拓扑/结构性**：betweenness / integration / connectivity / depth（Space Syntax 系）——GTG 直接把这当成跨城泛化的重要基石。
2. **出行条件与时空上下文**：出发时间、OD、轨迹统计量（Cardiff 把这些作为条件输入）。

如果你要用“文献支撑”来回答“特征是否足够”，最强的表达方式是：

* **对齐 SOTA 的特征族谱**（基础几何/语义 + 拓扑结构 + 时间/OD 条件），并做 **逐步加特征的 ablation**（像 GTG 那样）。

---

## 不确定点 2：候选集大小 tradeoff（~5 扩到 ~50–200 会不会掉性能？）

这件事在交通/路线选择里是经典问题：**choice set generation**。2024–2026 相关文献给你的核心结论是：

* 候选集越大，**覆盖率（把真实路线/真实选择包含进去）更好**，但会带来**噪声、相似候选过多、学习难度上升**；
* 高分辨率网络（更细的路网）会让这件事更严重。

### 2.1 交通路线选择领域：choice set generation 的最新证据（2024）

* 有工作专门对 **路径选择集生成算法做 benchmark**，用真实车辆轨迹数据对不同 OD 的候选生成算法进行比较（非常贴近你“从 ~5 到 ~50–200”的问题）。
* 在高分辨率网络上，有研究指出：基于 **K-shortest path** 的方案“容易实现”，但 **不太可能覆盖所有观测到的路线**（覆盖率问题）。

这两点可以用来支撑你的 tradeoff 表述：

> “扩大候选集能提升对真实路线的覆盖，但会显著增加模型判别难度与冗余噪声；尤其在高分辨率路网中，简单 k-shortest 往往覆盖不足。”

### 2.2 ML 方向：用深度学习来“算替代路线/候选路线”（2024）

PMLR 2024 有工作直接讨论 **Deep Learning-Based Alternative Route Computation**，并把它放在“替代路线计算”语境里与 k-shortest 等经典方法关联。

> 对你的意义：候选集并不一定只能靠图算法硬枚举；也可以“学习式生成候选 + 再判别/再排序”。

### 2.3 “动态候选集大小”有没有现成设计？

在你给的 2024–2026 片段里，我没看到一个**公认标准的“动态 K”框架**（比如明确提出根据熵/不确定性自适应扩大/缩小候选集的通用方案）。但可以从两条成熟主线“拼”出一个非常合理、论文也好写的设计：

**主线 A：路线选择的 choice set generation（先生成，再估计）**

* 用 choice-set 方法做 **粗筛**（保证覆盖率），再用你的模型做 **精选/排序**。

**主线 B：coarse-to-fine / multi-scale 生成（先粗规划，再细化）**

* Cardiff 的核心思想是把问题拆成 segment-level（粗）和 GPS-level（细），用粗层结果去约束细层生成。
* M-STAR（2025 arXiv）则是更“tokenization”风格的 coarse-to-fine：把轨迹投影到**不同分辨率的空间网格**，并做时间下采样，形成层级表示，然后用 Transformer 做“next-scale”预测。

> 你可以把“动态候选集”写成：
>
> * coarse level：小候选（~5–20），快速确定策略/走廊/区域/主干路段
> * fine level：只在 coarse 选定的走廊/子图内扩到 ~50–200
>   这样候选规模变大，但难度不会线性变大。

---

## 不确定点 3：Decision Point（DP）的“正确”定义是什么？

你提出的“高熵 + 道路等级变化”是一个很合理的工程启发，但如果你想要**认知/行为学文献支撑**，近两年的证据集中在两点：

1. **交叉口/岔路口（intersections）是决策的核心场景**
2. **地标（landmarks）会显著影响人类在决策点附近的导航表现与策略**

### 3.1 行为/认知证据：人在哪里做路线决策？

* 2025 的开放论文明确把 wayfinding 的关键组成之一描述为：在路径上遇到**交叉口时决定朝哪个方向继续**（研究关注老年人在此类决策上的变化）。
* 2024 的真实世界导航研究考察了**地标可视化风格**对导航任务表现的影响，说明“地标信息”会改变人类 wayfinding 行为。
* 还有 2025 ACM 工作关注对导航系统中的**视觉地标增强**需求与设计启示（尤其在 AR/辅助导航语境）。

**从这些证据出发，一个“更有文献背书”的 DP 定义**通常会包含：

* **结构性候选 DP**：图上真正存在分叉选择的点（交叉口/匝道口/多出边节点）
* **信息性候选 DP**：与地标/显著环境线索绑定的点（人会用地标做转向记忆与决策）
* **模型不确定性 DP**：高熵点（你提的）可以作为“信息性”补充，但不应成为唯一标准

### 3.2 SOTA coarse-to-fine 的 coarse level 怎么定义？

给你两个“2024–2026 可引用”的范式例子：

* **Cardiff（2025）**：把轨迹分解成 **road segment sequence（粗）+ GPS sequence（细）**，粗层用于承载转移结构与路网有效性，细层用于恢复微观位置细节。
* **M-STAR（2025）**：把轨迹映射到不同分辨率的空间网格，并配合时间下采样形成层级表示，再做逐级细化预测（“next-scale autoregressive prediction”）。

这两条路对应你 DP 定义的两种可能路线：

* DP = **路网结构上的关键点/关键路段**（segment / intersection driven）
* DP = **多尺度空间抽象单元**（region/grid token driven），再细化到路段/点

### 3.3 回到你的问题：高熵 + 道路等级变化够不够？

如果你要写得“逻辑分明且能落地”，我建议把 DP 定义拆成 3 条可检验准则（并把它们和文献对齐）：

1. **Choice existence（存在真实选择）**：DP 必须对应交叉口/分岔结构（行为学强调交叉口决策）。
2. **Cue richness（线索丰富）**：DP 附近是否有地标/显著线索（地标影响 wayfinding）。
3. **Model difficulty（模型确实困难）**：高熵作为补充，用来捕捉“结构上不明显但行为上分歧大”的点（这条更多是工程/学习理论动机）

---

## 不确定点 4：逐步 AR 选择 DP 是“最好”的方式吗？

文献趋势是：**AR 很自然，但“累积误差/暴露偏差”是长期生成的硬伤**；因此 2024–2026 大量工作在做：

* **non-AR（holistic）生成**（diffusion 等）
* **AR + diffusion 的混合**（用 AR 做规划/语义骨架，用 diffusion 做细节/多样性）
* **search/planning + learned cost**（绕开大候选分类）

### 4.1 AR 的已知问题（可以引用的表述）

Cardiff 的综述性描述非常直接：

* AR 方法适合离散数据，但存在 **受限采样与累积误差**，在连续结构（细粒度轨迹）上不够理想。

这句话可以直接成为你论文里“为什么考虑 non-AR / hybrid”的动机。

### 4.2 Traveller：AR-TempPlan + discrete diffusion（对你最贴的“AR vs diffusion”案例）

ScienceDirect 的摘要/引用信息给出了足够清晰的机制描述：

* Traveller 用 **AR-TempPlan** 捕捉时间规律，输出一个 **mask location sequence** 作为 temporal modes（时间规划信号）
* 然后用 **TravCond-Diff** 在空间上做生成：利用规划信号 + **home location（spatial anchor）** 通过**离散扩散过程**引导生成，从而提升轨迹的时空保真度与个体模式刻画。

> 这对你的“DP 逐步 AR”问题的启示：
> **把“先规划（时间/策略）”与“再落地（空间/路网约束）”拆开**，是 2025–2026 很强的一条路线。

### 4.3 Cardiff：非 AR 的 coarse-to-fine diffusion（离散路段序列也能扩散生成）

Cardiff 明确是 **segment-level（离散路段）→ latent diffusion → GPS-level 条件 diffusion** 的 cascaded 生成范式。

这说明：即便你的 DP/路段是离散的，也完全可以走 **“离散 token → 连续 latent → diffusion”** 的路线，而不是必须做逐步 AR 分类。

### 4.4 GTG：用“学习的偏好/代价 + 最短路搜索”绕开大候选分类

GTG 的生成不是“在巨大候选集合里分类”，而是：

* 用 road representation（含 Space Syntax 等）去建模 travel cost / 偏好
* 再用 **shortest path algorithm** 来生成路径

这对应你提出的替代方案：“预测路线类型/策略，再用 A* 生成路径”——GTG 可以作为非常强的近邻参考（只不过它用的是 shortest path/search 形式）。

### 4.5 离散序列上的 diffusion：有没有“成功案例”可作方法论支撑？

即使不在交通领域，离散序列 diffusion 在 2024–2026 已经有相当成熟的案例：

* ICLR 2025 的 Block Diffusion/BD3-LMs 把序列切成 block，在 block 内做离散 diffusion，从而在 AR 与 diffusion 之间插值，兼顾可控性与生成能力。

这可以作为你写“为什么 diffusion 可以用于离散 DP 序列”的方法论支撑。

### 4.6 Seed (WWW 2025)：“AR + diffusion”怎么结合？（我能给到的与缺口）

我目前只能拿到二手索引/摘要级信息（例如会议论文列表/聚合页），能确认它是一篇把 **Sequence 与 Diffusion 结合用于 road trajectory generation** 的 WWW 2025 工作。
但由于正文不可直接访问，我没法在不冒风险的前提下，给出它“sequence 部分候选集如何构建、AR 与 diffusion 如何对接”的精确实现细节（否则容易误读/臆测）。

**不过**：从 Cardiff/Traveller 已经能提炼出一条可靠的“AR+Diffusion 组合范式”，你可以先用作你的设计空间基线：

* **AR/序列模型做“规划信号”**（temporal modes / coarse skeleton / strategy）
* **Diffusion 做“细化与多样性”**（空间细节、约束满足、分布更平滑）

---

## 最后：把文献结论落到你 4 个不确定点的“可写进论文”的答案模板

### A) “特征是否足够？”——建议写法

* 先对齐 SOTA 特征族谱：

  * 基础特征（长度/类型/方向等）+ 拓扑结构（Space Syntax：integration、choice/betweenness…）+ 时间/OD 条件（departure time、OD、trajectory stats）。
* 再用 ablation 支撑：GTG 展示 Space Syntax 与 cost prediction 的重要性；Cardiff 展示坐标嵌入/表示压缩的重要性。

### B) “大候选集会不会掉性能？”——建议写法

* 用 route choice set generation 文献说明 tradeoff：覆盖率 vs 难度/冗余；高分辨率网络更难；k-shortest 不一定覆盖观测路线。
* 提出两阶段或层级化方案（coarse-to-fine）：Cardiff/M-STAR 提供可引用的层级生成范式。

### C) “DP 定义什么才正确？”——建议写法

* 引用 wayfinding 证据：交叉口决策是核心环节；地标影响导航任务与策略。
* 于是 DP 定义应包含：结构性分岔 + 地标/线索 + 高熵（补充项）。

### D) “AR vs 其他方式？”——建议写法

* 引用 AR 的局限（累积误差）。
* 给出 2025–2026 的替代路线：

  * AR+Diffusion（Traveller：AR-TempPlan 规划 + discrete diffusion 落地）。
  * non-AR diffusion（Cardiff：离散路段 latent diffusion + GPS 条件 diffusion）。
  * learned cost + search（GTG：road rep + shortest path）。
  * 方法论支撑：离散序列 diffusion 在 ICLR 2025 有成熟范式（Block Diffusion/BD3-LMs）。

---

## 2024–2026 推荐你优先精读的“最相关阅读清单”（按你的 4 类问题分组）

**特征/表征**

* GTG (AAAI 2025): Space Syntax + 基础路段特征 + SAGAT 关系特征；带消融。
* Cardiff (arXiv 2025): 路段 token + 中心点坐标嵌入 + OD/时间/统计条件；coarse-to-fine。

**候选集/choice set**

* Route choice set generation on high-resolution networks (2024): 指出 k-shortest 的覆盖问题等。
* Performance evaluation of path choice set generation (2024): benchmark 候选生成算法。
* Deep Learning-Based Alternative Route Computation (PMLR 2024): 学习式生成替代路线。

**决策点/行为学支撑**

* Intersection decision making / wayfinding（2025 open access）：交叉口决策是核心子任务。
* Landmark 与 wayfinding（2024–2025）：地标影响导航表现/系统设计。

**AR vs diffusion vs search**

* Traveller (online 2025; Inf. Fusion 2026): AR-TempPlan + discrete diffusion（home anchor + temporal modes guidance）。
* Block Diffusion / BD3-LMs (ICLR 2025): 离散序列 diffusion 的强方法论参考。
* GTG (AAAI 2025): learned preference/cost + shortest path。

---

如果你接下来希望把这份梳理直接变成“可写进论文的 related work 小节”，我可以按你论文结构给你一版：

* 先讲 **Route Choice / Decision Points（行为学 + 交通 choice set）**
* 再讲 **Road trajectory generation（AR / diffusion / hybrid / search）**
* 最后落到你的 method：**DP 定义 + 动态候选 + 特征消融实验设计**


下面把你新增的 **3 条检索需求（2024–2026）** 做一个“可直接落地到你当前路线/DP 建模”的文献梳理。为了让主线清晰，我按你关心的三个核心点来组织：**(1) 多模态 route 分布的 diffusion 建模**、**(2) 图连通性约束怎么做**、**(3) 不用 segment ID、只靠特征/embedding 的 segment 表示与解码**。
（备注：我尝试用工具对 PDF 做截图读取图表，但截图接口在当前环境报错；以下基于论文正文可解析文本进行梳理。）

---

## 0. 先给你一张“论文-问题”对照图（只列 2024–2026 代表作）

| 论文（年份）                                                  | 是否显式谈“多模态/多样性”                               | 连通/拓扑有效性怎么保证                                                                       | segment 表示（是否走连续 embedding）                                                                  |
| ------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **DiffPath**（2024/2025，ICLR25 OpenReview 标记为 withdrawn） | 明确强调 **long-tail + diverse path generation** | 提到 **clamping 机制**用于拓扑有效性 + 上下文一致性                                                 | “离散 path → 连续 latent → diffusion → 再解码回离散”端到端学习 embedding                                    |
| **Cardiff**（2025, arXiv:2507.13366）                     | 更偏“高保真+一致性+隐私”，多模态主要靠采样（不太“显式”用多样性术语）        | **Spatial Validity Loss**（邻接矩阵）+ **后期才加约束**（adaptive strategy）+ 解码 **beam search** | 先用 **segment autoencoder + latent compression** 得 latent，再生成 latent，最后 **AR 解码回 segment 序列** |
| **GDP**（ICLR 2024）                                      | 以“conditional sampling”做端到端路径规划，本质支持多解采样     | 核心卖点：**diffusion process 显式融入 road network graph constraints**                     | 在图上对“顶点序列”做扩散（categorical diffusion 思路）                                                      |
| **ControlTraj**（KDD 2024）                               | 明确把“人类活动多样性/不可预测”作为动机；目标是“可控、多条”             | 用 **RoadMAE** 提供拓扑约束条件引导 GeoUNet（生成结果更贴合路网）                                        | 重点是 **road segment embedding（RoadMAE）作为条件**，生成 GPS 轨迹                                        |
| **Diff-RNTraj**（2024, TKDE/ArXiv）                       | 更强调“生成在路网约束上 + 带道路信息”，多模态主要靠采样               | 用 **空间有效性损失**增强 on-road/validity；表示本身也“天然在路上”                                      | **segment(离散)+moving rate(连续)** → 先 embed 到连续 → diffusion → decoder 回 hybrid                 |
| **Seed**（WWW 2025）                                      | 明确说要平衡 **diversity & regularity**            | （你前面关心的 AR+diffusion 结合，它属于“多模态+可控性”的路线）                                           | 序列模型 + diffusion 组合，用于路网轨迹生成                                                                 |

上表每个单元格的论据出处我在后文分点展开并逐条引用。

---

## 1) Diffusion for Multi-modal Route Generation：有没有显式讨论“多模态 route/trajectory 分布”？

### 1.1 明确“把多样性/多模态当成目标”的 route/path diffusion 工作

**(A) DiffPath：把“长尾 segment 导致多样性不足”当成核心挑战之一**
DiffPath 把“路段长尾分布”与“多样性”直接挂钩：它在贡献里明确写了 **custom loss 处理 long-tail，确保 diverse path generation**，同时还做了 **positional embeddings + clamping** 来处理“拓扑有效性/上下文一致性”。
这类写法很契合你说的“corridor diversity / 多模态 route 分布”：它至少承认**多解**不是“采样一下自然就有”，而是需要在训练目标上对抗 long-tail 的“模式坍塌/主干路段垄断”。

**(B) Seed（WWW 2025）：直接把 diversity 作为评价维度之一**
Seed 的 OpenReview 摘要里明确说它要平衡 **diversity 与 regularity**，并声称在 accuracy、diversity、regularity 上达到 SOTA。
如果你要写“为什么扩散模型适合多模态路线分布”，Seed 这种表述是非常好的引用支点（因为它把“多样性”写进了主张里）。

**(C) ControlTraj（KDD 2024）：把人类出行的“多样性/不可预测”作为 diffusion 动机，并且目标就是“可控地产生多条”**
ControlTraj 的引言/摘要非常明确：现有方法难在“inherent diversity and unpredictability”，它提出 topology-constrained diffusion，并且问题定义也强调生成器要产出多条轨迹并对齐给定路网拓扑约束。
这可以作为你“多模态是 mobility 的本质属性之一，模型需要显式支持一对多”的背景引用。

---

### 1.2 “多模态主要靠采样 + 分布拟合”的路线（不一定把 diversity 写在标题里）

**(A) GDP（ICLR 2024）：把路径规划写成 conditional sampling，本质支持多解**
GDP 把 end-to-end path planning 表述为：先学无条件路径分布，再把 OD 作为条件做采样，且强调 bypass search-based framework。
这类方法天然可以“采样多条候选路径”，因此在“多模态 route 分布”上是成立的，只是它更关注“路径分布与历史数据一致”，而不是专门做 diversity 指标。

**(B) Cardiff（2025）：两阶段扩散（segment-level → GPS-level），强调 consistency/validity/隐私；多模态更多来自 diffusion sampling**
Cardiff 把 mobility 的层级结构（segment-level 与 GPS-level）和 cascaded diffusion 对齐：先生成“road-network-consistent segment trajectories”，再细化到 GPS。
它在“多模态”上不如 DiffPath/Seed 那么“口径明确”，但**扩散模型采样本身**就是生成多个可能解的机制；更关键的是它把“有效性（road-network-consistent）”放到第一优先级上（这与你第二条需求高度耦合，见下文 2.2）。

**(C) Traveller（Information Fusion 2025）：用 AR 学 temporal modes，再用离散 diffusion 做空间生成**
Traveller 的摘要里提到两部分：AR-TempPlan 用于生成 mask location sequence（可理解为 temporal mode / 计划信号），TravCond-Diff 在空间层面用 discrete diffusion 生成 trajectory（用 planning signal + home location anchor 来 guide）。
从结构上看，它是“先产生一个高层 plan/模式，再生成细节轨迹”，这通常会强化多模态（不同 plan → 不同 corridor/route family）。但它的摘要并没有直接说“corridor diversity/多样性指标”，所以更像“结构上支持多模态”。

---

### 1.3 你问的“Cardiff/Traveller 是否**显式**处理多模态？”

* **Cardiff**：文本上更强调的是“层级一致性 + 道路网络一致性 + 隐私评估”，多模态更像 diffusion 的自然产物，而不是用专门的 diversity 目标来显式推动。
* **Traveller**：结构上“AR 产生 temporal plan/mode + diffusion 产生空间轨迹”很像显式把多模态拆成“模式（plan）→ 轨迹”，但从摘要可见它更多强调“travel-pattern-aware / guide generation”，没有直接把“多样性”作为主打指标来讲。

如果你论文里要写得严谨：可以表述为

> “这些工作提供了**生成多解**的机制（sampling / plan-to-trajectory），但只有部分工作（如 DiffPath、Seed）在目标函数或主张中**明确把 diversity 当成一等公民**。”

---

### 1.4 “有没有用 Classifier-Free Guidance 控制 corridor 选择的工作？”

我没有在“route corridor 选择”这个**特定表述**上看到直接命中的论文，但有两条非常关键、可直接迁移的证据链：

**(A) CFG 已经被用于“轨迹/序列”生成的 guided sampling**
NeurIPS 2024 的 *Guided Trajectory Generation with Diffusion Models for Offline Model-based Optimization* 明确写了：采样时使用 **classifier-free guidance** 来生成 trajectories，并通过 guidance/conditioning 去探索 high-scoring regions。
这说明：**CFG 不只是图像技巧，已经在“trajectory”域被当成标准可控采样手段**。把“score”换成“corridor 偏好/策略偏好/道路等级偏好/避收费”等，你就能得到 corridor-level steering 的一个有文献依据的路线。

**(B) 离散 diffusion 的 CFG/Guidance 机制在 2024 已经有专门讨论**
*Simple Guidance Mechanisms for Discrete Diffusion Models*（2024）专门讨论离散扩散的 controllable generation，并给出 classifier-free / classifier-based guidance 的推导与经验结果。
你的 corridor 选择如果最终落到“离散 DP/segment token 序列”的生成，这篇可以作为“离散域也能做 CFG”最直接的 methodological backing。

**落地到你的 corridor 控制**：

* 把“corridor/路线类型/策略”当成 condition (y)（离散标签或连续 embedding）。
* 训练时做 condition dropout（得到 conditional 与 unconditional score），推理时用 CFG scale (w) 在“更贴合 corridor”与“更多样探索”之间调参。该做法在 trajectory guidance 与 discrete guidance 两条链路上都有现成论文支撑。

---

## 2) 图连通性约束：diffusion 生成的 segment sequence 如何保证“在图上连通/有效”？

这里我建议你把现有方法分成 3 类：**(i) 过程级（diffusion process）就遵守图结构**、**(ii) 损失/约束项把“连通性”作为训练目标**、**(iii) 解码/采样时做硬约束或修正**。

### 2.1 过程级约束：GDP（ICLR 2024）是最“硬核”的代表

GDP 的核心贡献之一就是：**设计了一种 diffusion process，把 road network 的 graph constraints 融入扩散过程**，并将路径规划视为 OD 条件下的生成。
它对“路径”的定义也是直接要求相邻顶点必须相邻（graph adjacency）。

**对你有用的点**：
如果你担心“生成出来的序列大量断边”，GDP 这类思路是从机制上把“断边”变成低概率事件（甚至过程上就不允许），而不是事后修补。

---

### 2.2 训练目标级约束：Cardiff 的 Spatial Validity Loss + “后期才施加约束”的策略

你问得非常具体：“Cardiff 的 segment diffusion 如何保证序列在图上连通？”——Cardiff 在 coarse stage 给了非常清晰的答案：

* 它定义了 **Spatial Validity Loss**，用 road network adjacency matrix (A) 表示两个 segment 是否相邻，并据此惩罚生成序列中“不相邻的连续 segment”。
* 同时它还提出一种 **adaptive physics-informed** 的策略：**不是从一开始就强加这些约束，而是在后期 denoising steps 才施加**，避免早期过强约束干扰整体分布学习。

**这正好回答你“diffusion 过程怎么加 graph constraint”的问题**：Cardiff 选择的是“损失 + 时间调度（late constraint）”这一类，而不是 GDP 那种“过程级重定义”。

---

### 2.3 条件引导/结构编码：ControlTraj 用 RoadMAE 把拓扑约束变成 conditional guidance

ControlTraj 的做法更像“把 topology 变成条件信息”，再用 conditional diffusion 去贴合：

* 它明确说将 road network topology 的 structural constraints 用来 guide geographical outcomes，并提出 road segment autoencoder 去抽取 road segment embedding，最后把 road embedding + trip attributes 融合进 GeoUNet。
* 它的框架图说明也很直白：**RoadMAE 基于 road segments 的 topology constraints 编码 road embedding**，然后在 diffusion 里用 geographic attention 注入。

这类路线的优点是：你可以把“连通性/可达性/道路等级规则”编码进条件向量里，让模型“学会沿着路网生成”，而不是只靠后处理。

---

### 2.4 解码/采样时硬约束或修正：DiffPath 的 clamping + Cardiff 的 beam search + Diff-RNTraj 的“表示即有效”

**(A) DiffPath：clamping mechanism for topological validity**
DiffPath 在贡献里直接写了它用 **clamping mechanism** 来处理“topological validity and contextual coherence”，并且还配合 long-tail loss 来支持 diverse generation。
虽然你如果要复现细节需要细读其方法部分，但至少它在“采样/生成阶段对拓扑做硬处理”这一点上，给了明确的文本证据。

**(B) Cardiff：latent → AR 解码 + beam search**
Cardiff 的 coarse stage 会先生成 latent，然后用 AR transformer token-by-token 解码 segment 序列，并使用 **beam search**。
beam search 在工程上非常适合**把“只允许扩展到邻接 segment”做成 decoding-time mask**（Cardiff 文本里没把“mask”写成显式公式，但它给了 beam search 这一可操作抓手）。

**(C) Diff-RNTraj：表示层面把“在路上”变成先验**
Diff-RNTraj 把轨迹点定义为 **(离散 road segment, 连续 moving rate)** 的 hybrid 形式；训练时先把 hybrid embed 到连续空间做 diffusion，采样后再用 decoder 映射回 hybrid，并引入新 loss 增强 spatial validity。
这一类表示天生就比“纯经纬度扩散”更容易保证 on-road（因为输出空间本身就贴着路网）。

---

## 3) Feature-based Segment Representation：不用 segment ID，只用特征/连续 embedding，怎么编码与解码？

你这里其实问了两件事：

1. **有没有“先把 segment/path 编码成连续向量，再在连续空间生成”的工作？**
2. **生成后如何从连续 embedding 恢复到离散 segment？**

答案是：2024–2025 这条线已经非常明确，并且不同论文给了不同“解码回离散”的范式。

---

### 3.1 “离散 path/segment → 连续 latent → diffusion 生成 → 解码回离散”的代表：DiffPath

DiffPath 的方法段落把这件事写得非常规范：

* 它定义 embedding 函数 **EMB(vi)**，把离散 path node（可理解为 intersection 或 road segment）映射到 (\mathbb{R}^d)，得到 path 的连续表示 (\in \mathbb{R}^{l \times d})。
* 它强调 embedding 与 diffusion 参数是 **end-to-end joint learning**，不是固定 embedding。
* 在 reverse 过程中，它把 latent 逐步 denoise，然后 **在每个位置选择最可能的 road segment 来解码回离散 path**。

> 对你第三条需求“解码怎么做”：DiffPath 给的是最直接的一类——**position-wise 的离散选择（类似分类/argmax）**。

---

### 3.2 “先压缩成 latent，再用 AR decoder 恢复离散序列”的代表：Cardiff

Cardiff 的 coarse stage 更像你现在做 DP/segment 序列时会喜欢的结构：

* 它用 transformer-based segment autoencoder 把 segment-level trajectory 压缩成 latent（Perceiver 风格 latent compression），然后 diffusion 在 latent space 生成；最后再解码回 segment 序列。
* 解码阶段是 **AR transformer token-by-token**，并用 **beam search** 提高恢复质量。
* 同时，它还把 road segment 的 **center-point coordinates** 加到 token embeddings 中，用以注入空间信息。

> 对你第三条需求“只用特征，不用 ID”：Cardiff 本质还是把 segment 当 token，但它明确在 embedding 里注入了几何位置；而且它的“latent→AR 解码”范式非常适合你把“segment 特征编码器（GNN/MLP）输出 embedding”替换掉纯 ID embedding。

---

### 3.3 “road embedding 作为条件输入”的代表：ControlTraj（更强调跨城市泛化）

如果你的目标是 **不想依赖 segment ID vocabulary（尤其跨城市）**，ControlTraj 的思路非常值得你借鉴：

* 它提出 **road segment autoencoder（RoadMAE）** 提取 fine-grained road segment embedding，并强调 road embedding 是基于 topology constraints 编码出来的；再与 trip attributes 拼接作为 diffusion 的 conditional guidance。
* 它还强调可迁移到新城市（框架图/引言里就有“new city without retraining”的叙述语境）。

> 对你“Feature-based Segment Representation”的启发：
> **先用 RoadMAE/MAE/GNN 之类把 segment 的 topology+geometry+attributes 学成连续 embedding**，再把生成建模放到 embedding/latent 空间里。这样 segment ID 只作为索引，不作为语义载体，跨城更自然。

---

### 3.4 “混合表示（离散 segment + 连续速率）”的代表：Diff-RNTraj

Diff-RNTraj 的关键点是：**把‘在路上’和‘路上细节’作为数据结构的一部分**：

* 每个点是（segment，moving rate），是 hybrid；模型用 pre-training 把 hybrid 嵌入到连续表示，再用 continuous diffusion 生成，最后用 decoder 映射回 hybrid，并加 spatial validity loss。

> 对你“解码回离散”的启发：
> 你可以把“解码”设计成一个显式 decoder（连续→离散+连续），而不是只做 nearest neighbor 或 argmax，这在处理“连通性/物理可行性/速度”等约束时更灵活。

---

## 4) 给你可直接写进方案/实验章节的“结论型建议”（与三条需求一一对应）

### 4.1 多模态 route：建议用“训练端 long-tail + 推理端 guidance”双管齐下

* 训练端：参考 DiffPath 对 long-tail 的处理，至少要承认“主干路段垄断会吞掉小众 corridor”这一风险，并在 loss 或采样策略上做补偿。
* 推理端：如果你要“可控地选 corridor”，不要只靠随机种子；可以借鉴 trajectory-guidance 工作用 CFG 做 steer（NeurIPS 2024），以及离散 diffusion 的 guidance 推导（2024）。

### 4.2 连通性：优先选“机制内约束”，其次“late-stage validity”，最后“后处理”

* 如果你最担心 invalid transition：GDP 是“过程级约束”的代表（更硬）。
* 如果你更关心生成质量、怕过强约束伤分布：Cardiff 的 **validity loss + 后期施加** 是非常实用的折中。
* 还可以叠加 decode-time 的 beam/mask（Cardiff 给了 beam search 的抓手）。

### 4.3 feature-only segment：别纠结“完全不用 ID”，核心是“语义来自特征/拓扑，而不是 ID embedding 表”

* 想跨城泛化：走 ControlTraj 的 RoadMAE/topology-aware embedding 思路，把路段特征/拓扑学成 embedding 再做生成。
* 想更像“生成 segment 序列”：用 Cardiff 的 latent→AR 解码；想更像“并行生成”：用 DiffPath 的 position-wise 解码（或再加连通 mask）。

---

如果你愿意把你当前的“segment 特征集合”（你提到的相对位置、道路等级、方向角等）和你打算的“corridor/策略条件”定义（比如 region sequence、road class pattern、收费/拥堵偏好）贴出来，我可以在不再扩展检索的前提下，直接用上面这些文献里的**范式**帮你把：

* “corridor 条件怎么定义成可训练的 y（用于 CFG/离散 guidance）”、
* “连通 mask/validity loss 放在 diffusion 的哪个 stage/step”
* “连续 embedding→离散 segment 的解码器该用 DiffPath 式还是 Cardiff 式”
  三个点给出一套更具体的架构草案与实验矩阵。

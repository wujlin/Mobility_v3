下面给你一份**按“主线—分支—代表作—可借鉴点/空白”**组织的文献梳理，覆盖你给的三组 Query（时间窗尽量锁定在 2023–2025/2024–2025，并在必要处补 1–2 篇更早的“承上启下”工作作为对照）。

---

## 总主线：把 Route Generation 还原成“语义规划 → 走廊/路径决策 → 连续轨迹落地”

你们三个 Query 其实可以用同一条主线串起来：

1. **城市语义（POI/land use/场景）**提供“为什么走这条路”的解释变量与约束；
2. **路线生成天然是离散+连续混合**：

   * 离散：选哪条走廊/哪一串路段/哪类语义偏好（安全、景观、商业、避堵…）
   * 连续：在道路拓扑/几何约束下生成可行轨迹（GPS 点、路段序列、速度分布等）
3. **Trajectory foundation model**提供“可迁移、可泛化”的底座表征，支撑长时域（long-horizon）与语义条件（semantic conditioning），并能作为生成模型的 encoder/backbone 直接接入生成框架（route generation）。

下面按你们三条 Query 展开。

---

# Query 1：Route Choice/Generation 的 Semantic Modeling（2023–2025）

### Q1-1：现有 route generation/route recommendation 如何用 POI / land use / urban semantics？

我建议按“语义注入的层级”来梳理（从弱到强）：

## 1) 语义作为“特征/上下文”注入：POI/文本/图像/天气等多模态增强

这条路线更偏 **route recommendation**，核心是：轨迹/路网之外，把“城市语义”作为额外 modality 输入，学更贴近用户偏好的路径分布。

* **Survey 视角（2024）**：Zhang 等的 route recommendation survey 明确把“multi-modal approaches”作为重要分支，强调除了轨迹与路网外，还会融合天气、POI 图像/文本/音频等新模态，并举例用 **Google Street View 城市景观**来做“多样化场景”的路线推荐。

  * 这给你们一个很实用的**语义来源清单**：POI 图像/文本、语音描述、天气、不同交通网络等，都可以作为“urban semantics”的可操作数据源。

* **POI 表征增强（2025, AAAI）**：POI-Enhancer 用 LLM 通过 prompt 抽取 POI 相关语义，再用对齐/融合模块把语义注入到 POI embedding，提升下游任务表现。

  * 对 route generation 的意义：如果你们希望“路线偏好”能被解释为对 POI/功能区的偏好，那么 **POI embedding 的语义质量**会直接决定上层“语义规划”的效果。

* **POI → Land use/功能区语义（2025, ORNL）**：有工作把 AOI 内 POI 的空间分布与语义属性转成高维 embedding，用于多尺度、多粒度的 land use 表征。

  * 这类“POI 驱动 land use embedding”可以当作你们语义走廊（corridor）/网格语义的底座。

> 小结：这一路线的典型形态是 **“把 POI/land use 变成 embedding → 和轨迹/路网特征融合 → 学一个偏好分布/评分函数”**。它能做个性化、场景化推荐，但通常还没把“走廊选择”显式建模成一个可解释的语义决策过程。

## 2) 语义作为“结构化知识”注入：KG/语义图谱辅助路径

这一路线的特点是：把路网/路线与外部知识（POI、关系、类型、语义约束）做成图结构，再在图上做推理/推荐。

* **RouteKG（2023）**：以知识图谱视角来做 route recommendation（把路线相关要素组织成 KG 以便嵌入与推理）。

  * 适合回答“为什么推荐这条路”（可解释关系链），也便于把“POI/land use”接入为图谱实体/属性。

## 3) 语义作为“决策机制”注入：把 route choice 显式建模为语义驱动的序列决策

这是你们 Q1 的关键：不仅把语义当特征，而是把**“每一步怎么选路”**视为语义决策过程。

* **CORE（2025）= 目前最贴近你们“corridor choice as semantic decision”的工作之一**
  这篇的贡献点很“对题”，可以作为 Query 1 的主轴论文：

  **(a) 多粒度环境语义建模（fine + coarse）**：

  * coarse-grained：用 POI 类别在网格上的分布建“功能热点”，还显式用卷积核刻画热点对邻域的“spillover effect”。([arXiv][1])
  * 然后把这些多粒度环境语义与路段基础属性（road type、长度、入/出度等）用 gating 融合，形成**context-aware road segment representation**。([arXiv][1])

  **(b) 把 route choice 写成“从邻接候选集合里选下一段路”的序列决策**：

  * 给定轨迹路段序列，每一步从邻接集合里选下一个路段；
  * 同时引入“历史转移概率”“面向目的地的方向偏差”等导航因素作为 adjacent route context。([arXiv][1])

  **(c) 还能对标 generation / path ranking**：它在实验中把 trajectory generation 作为评测项之一，并在 baseline 列表里清楚区分了多种生成范式（例如 **TS‑TrajGen：two-stage GAN + A*；STEGA：semantic-aware graph；HOSER：多层路网语义融合**）。([arXiv][1])

> 从“你们的问题表述”看：CORE 的 Route Choice Encoder 已经非常接近“把 corridor choice 显式建模为 semantic decision”，因为它不是只给 route 打分，而是在路网邻接约束下建模每一步的选择，并且选择依据里包含了来自 POI/功能热点的环境语义。

## 4) LLM/Agent 路线：把“语义决策”外显为可交互推理

这条路线在 2025 明显升温：用 LLM/Agent 把“路线偏好、场景语义、规则约束”做成可对话/可解释的规划。

* **PathGPT（2025）**：用大语言模型做个性化路径规划。
* **PathGen‑LLM（2025）**：强调“交互式”的 LLM 路线生成框架。
* **Agentic Vehicular Routing with Semantic Context（2025）**：标题就直接点出“semantic context”，属于把语义上下文引入车辆路径规划的代表。

> 这类工作通常会把“语义”显式体现在输入与中间推理链里（例如偏好、场景、规则），但其不足也明显：可复现评测标准、对路网拓扑/交通规则的严格可行性保证、以及大规模泛化能力，往往仍需要和传统/学习式约束模块结合。

---

### Q1-2：有没有工作 **explicitly** 把 corridor choice 建模为 semantic decision？

可以把答案分成“强显式”和“弱显式”两档：

* **强显式（最接近你们问法）**：CORE 把 route choice 建模为“每一步从邻接候选中选下一路段”的序列决策，并且路段表示里融合了 POI/功能热点等环境语义，属于把“选择机制”显式写出来的范式。([arXiv][1])
* **弱显式（语义推理显式、但决策结构未必严格形式化）**：PathGPT/PathGen‑LLM/Agentic routing 把语义上下文作为规划依据，通过 LLM 推理驱动路线生成或交互式修改。

---

### Query 1 的关键空白（你们可以对齐后续研究设计）

1. **“语义走廊/功能走廊”级别的 choice set & 评测**：多数工作还停留在路段序列/路径打分；把 corridor 当作可解释的语义对象（商业走廊、景观走廊、避险走廊等）并建立统一 benchmark 的工作仍少。
2. **语义与可行性耦合**：语义偏好往往和交通规则/拓扑约束/实时交通冲突，如何在生成阶段“硬约束可行，软约束语义”是核心工程难点。
3. **land use 的使用仍偏“间接”**：更多是用 POI/网格热点去近似 land use（CORE 就是典型），直接用遥感 land use/街景语义做可控走廊选择的工作还不成体系。

---

# Query 2：Multi‑modal Generative Models 的 Discrete‑Continuous 混合（2024–2025）

你们的问题本质是：**怎么让生成模型同时做“离散决策（选什么）”和“连续生成（怎么走）”？** 这在 diffusion/flow matching 框架里有两条主流路线：

---

## 2.1 “同一生成过程里同时建模离散与连续”——Hybrid diffusion / Hybrid state space

### CANDI（2025）是一个很标准的“离散‑连续混合扩散”代表

* CANDI 讨论了“连续扩散直接用于离散 token 会掉点”的原因，并提出把 **离散 identity corruption** 与 **连续几何退化**解耦，形成 hybrid discrete‑continuous diffusion。
* 它还强调这样做能让离散空间利用连续扩散的梯度信息，并支持基于分类器的 controllable generation。

> 对路线生成的映射：
>
> * 离散层：走廊/路段 token、功能区 token、或者“决策点选择”token；
> * 连续层：在路网上的几何轨迹、速度曲线、微观动作。
>   这类 hybrid diffusion 给的是一个“统一扩散过程”思路：你不一定要 two-stage，也可以把两类变量放在一个联合生成过程里（代价是建模复杂度更高）。

---

## 2.2 “先解决离散生成，再把离散结果作为条件驱动连续生成”——Discrete generative backbone + continuous executor

### Discrete Flow Matching（2024）提供了“离散生成”的非自回归范式

* Discrete Flow Matching 把 flow matching 推到离散域：定义概率路径、用 learned posterior（如 x‑prediction / ε‑prediction）进行采样，并且强调可以非自回归地产生高维离散数据。

> 对路线生成的映射：
>
> * 第一阶段可以用离散流/离散扩散生成：**路段序列、走廊序列、网格序列、或“语义计划 token”**；
> * 第二阶段再用连续扩散/flow matching 生成细化轨迹（或做 map-constrained refinement）。

---

## 2.3 你们问的 “semantic planning + continuous execution” two-stage：2024–2025 的典型答案是“层级扩散策略/规划器”

这一块在机器人/长时域规划里非常成熟，概念可以直接迁移到 route generation：

### Hierarchical Diffusion Policy (HDP, CVPR 2024)：明确的 two-stage（高层规划 + 低层扩散执行）

* HDP 把策略分解成：

  * 高层 task-planning：预测较远的 next-best end-effector pose；
  * 低层 goal-conditioned diffusion policy：生成具体运动轨迹；
    并引入 kinematics-aware 的 RK‑Diffuser 来满足运动学约束。

> 对应到路线生成：
>
> * 高层（semantic planning）：先生成“下一段走廊/子目标/功能区/关键 waypoint”（可以是离散或低维连续）；
> * 低层（continuous execution）：在路网拓扑与几何约束下生成可行轨迹。

### CHD（2025）：进一步解决“层级规划上下层不耦合”的痛点

* CHD 指出现有层级扩散在 long-horizon 下失败的原因之一是高层子目标与低层轨迹生成**耦合太弱**，于是提出在统一扩散过程中**联合生成高层 sub-goals 与低层 trajectories**，并用共享 classifier 把低层反馈传回上层，让子目标在采样过程中自我纠正。

> 对路线生成的启示非常直接：
> 如果你们的“走廊选择（语义/离散）”与“连续轨迹（几何/可行性）”相互影响很强，那么 CHD 这种“边采样边纠错”的耦合式层级扩散，可能比“先定走廊再生成轨迹”的硬 two-stage 更稳。

---

# Query 3：Trajectory Foundation Models 最新进展（UniTraj 之后）

这里我按“**是否真正在做 foundation model** + **是否评测 generation（而非纯 prediction）**”来梳理。

---

## 3.1 UniTraj（2024/2025版本更新）之后：有哪些新/更通用的 trajectory foundation 模型？

### GenMove（2025）：masked conditional diffusion 的“一模多任务”框架（含 generation）

* GenMove 明确瞄准“不同任务格式不统一、条件复杂”两大问题，用 mask condition 统一任务格式，并用 contextual trajectory embeddings（含时空特征与用户偏好）+ classifier‑free guidance 来适配不同条件；报告在 generation 任务上最高提升超过 13%。

### MoveGPT（2025）：把 mobility foundation model 做到“大规模 + 多城市迁移”的 MoE 路线

* MoveGPT 提出用 **unified location encoder**把不同城市/不同位置映射到共享语义空间，并用 **spatially-aware mixture-of-experts Transformer**做可扩展的 mobility foundation model，用于跨城市/跨任务迁移（paper 标题和摘要主张其通用性与可扩展性）。

> 这类 MoE/跨域对齐的价值：对你们“route generation + urban semantics”来说，最大痛点往往是 **跨城市语义不对齐**（POI 稀疏、类别体系不同、路网形态差异）。MoveGPT 的设计动机就是解决“空间异质性下的统一建模”。

（补充：在 UniTraj 之前还有 TrajFM（2024）等“foundation”取向工作，但你们问题是 *UniTraj 之后*，所以这里重点放 2025 的新进展。）

---

## 3.2 它们如何处理 long-horizon？

从你们关心的“路线生成”角度，long-horizon 主要有两种难点：**序列太长**与**误差累积**。对应两类策略：

1. **通过预训练/表示学习提升“长序列可压缩性”**

   * UniTraj 的思路是用大规模轨迹数据做通用预训练，并在 generation 上通过“接入下游生成框架”体现价值（见下一节）。
   * MoveGPT/GenMove 则更偏“任务统一 + 条件控制/迁移”，把长时域生成当作统一框架下的一个任务类型。

2. **通过层级生成减少误差累积（规划—执行分解）**

   * 这其实与 Query 2 强相关：HDP（two-stage）与 CHD（耦合层级）都属于面向 long-horizon 的生成式规划范式。
   * 对 route generation：把全程拆成子走廊/子目标段，可以显著降低一次生成的难度与累积漂移。

---

## 3.3 它们如何做 semantic conditioning？

这里给你一个“从弱到强”的语义条件方式谱系（并标注对应代表作）：

1. **语义作为条件 embedding（POI/偏好/上下文 → 向量）**

   * GenMove：用历史轨迹得到 contextual embedding，显式包含时空特征与用户偏好，再用 classifier-free guidance 控制输出。

2. **语义作为“环境感知模块”，直接影响路段/网格表示**

   * CORE：用 POI 类别的空间分布建 coarse-grained “功能热点”，再与路段属性融合成 context-aware 路段表示；并在 route choice encoder 里用这些表示驱动“下一段选择”。([arXiv][1])

3. **LLM 注入显式语义知识**

   * POI-Enhancer：LLM 提取 POI 语义并融合进 POI embedding。
   * CORE 也出现 LLM-based description + text embedding 的处理链条（把环境语义“文本化/可解释化”再嵌入）。([arXiv][1])
   * PathGPT/PathGen-LLM/Agentic routing：把语义偏好通过自然语言显式表达，形成交互式规划。

---

## 3.4 它们在 route generation（not prediction）上的表现如何？

这里我建议你们把“route generation”拆成两类评测口径，因为很多论文里“generation”并不等价于“OD 路线生成”：

### A) 生成“轨迹/路段序列”本身（synthetic trajectory / controllable generation）

* **ControlTraj（KDD 2024 / arXiv 2024）**：用 topology-constrained diffusion 来生成高保真可控轨迹，并通过 RoadMAE 学 road segment embedding、GeoUNet 做地理去噪生成，强调能在不同真实数据设置下生成可控轨迹并适配未见地理环境。

  > 这类工作非常贴近“route generation”中的“落地生成（continuous execution）”，因为它把路网拓扑作为硬结构注入扩散过程。

* **UniTraj 的 generation 评测方式**：它并不是自己从零发明一个生成器，而是把 ControlTraj 当作下游生成框架，把 ControlTraj 的 road segment 抽取模块（RoadMAE）替换为 UniTraj encoder，观察生成指标变化：

  * Chengdu 上 density error 从 0.0039 降到 0.0037（约 5.1% 降幅）；
  * 迁移到 Xi’an（不重训）时，UniTraj-enhanced 的 density error 0.0152，优于 baseline ControlTraj 的 0.0171。

  > 这很关键：说明 foundation model 的价值可以体现为 **“给生成框架提供更通用的路段/轨迹表征”，从而提升跨城生成迁移**。

* **GenMove（2025）**：明确把 generation 作为目标任务之一，并报告在 generation 任务上最高提升超过 13%。

### B) 真正意义上的“OD 路线生成/路径规划式生成”（更接近你们的 route generation 直觉）

* 严格来说，学术界对这类评测还**不够统一**：很多论文的 generation 更偏“符合统计分布的合成轨迹”，而不是“给定 OD + 偏好 + 约束，生成一条可解释走廊与可行路径”。
* 目前最贴近“把 choice 过程写清楚”的，反而是 CORE 这种把每一步的候选集合与决策因素显式化的 route choice 建模（它也做了 trajectory generation 与 path ranking 的评测项）。([arXiv][1])
* LLM/Agent 的路线（PathGPT/PathGen‑LLM/Agentic routing）在“语义可交互规划”上更像 OD 路线生成，但在可行性约束与标准化指标上仍需补齐。

---

# 一页式对照表：你们三条 Query 的“核心论文—可复用模块—对应问题”

| 方向                                    | 代表工作（年份）                     | 你们关心的点                                                | 可复用的“模块化思路”                                           |
| ------------------------------------- | ---------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| 语义路线选择                                | CORE（2025）([arXiv][1])       | 显式 route choice；POI/功能热点语义；可做 generation/path ranking | 多粒度环境语义 → 路段表示 → 邻接候选集合上的决策编码                         |
| 语义增强 POI embedding                    | POI‑Enhancer（2025）           | POI embedding 语义化                                     | LLM 抽取语义 → 对齐/融合 → 作为 routing/conditioning 的“语义底座”    |
| 多模态 route recommendation              | Survey（2024）                 | POI 图像/文本、街景、天气等                                      | “语义数据源地图” + 方法分类（可做 related work 框架）                  |
| 离散生成（planning token）                  | Discrete Flow Matching（2024） | 离散决策生成                                                | 非自回归离散生成，可用来生成走廊/路段 token                             |
| 混合离散-连续扩散                             | CANDI（2025）                  | 离散+连续混合                                               | 解耦离散腐蚀与连续噪声，统一生成过程                                    |
| Two-stage 规划-执行                       | HDP（CVPR 2024）               | semantic planning + continuous execution              | 高层子目标/姿态 → 低层扩散轨迹（约束可行性）                              |
| Long-horizon 耦合层级扩散                   | CHD（2025）                    | 长时域；上下层耦合                                             | 联合扩散 + 低层反馈修正高层子目标                                    |
| Route/trajectory generation 框架        | ControlTraj（2024）            | 路网拓扑约束下扩散生成                                           | RoadMAE + 拓扑约束 GeoUNet；可控生成                           |
| Trajectory foundation + generation 迁移 | UniTraj（2024/2025）           | foundation + generation 迁移评测                          | 用 foundation encoder 替换生成框架组件，提高跨城 generation 指标      |
| 通用 mobility foundation（跨城/跨任务）        | MoveGPT（2025）                | scaling / transfer / 统一空间语义                           | 统一 location encoder + spatially-aware MoE Transformer |

---

# 给你们的“写作建议”：怎么把这三条 Query 写成一条清晰 related work 主线

如果你们要写成论文 related work / 文献综述，推荐用这三段式结构（每段 1–2 页，逻辑非常顺）：

1. **Semantic Modeling for Route Choice/Generation**：

   * 先按 survey 的分类引入多模态语义（POI 图像/文本/街景/天气）
   * 再引出“语义→决策”的显式建模：以 CORE 为代表（多粒度语义 + 邻接集合决策）([arXiv][1])
   * 最后补 LLM/Agent 的交互式语义规划路线（PathGPT/PathGen‑LLM/Agentic routing）。

2. **Discrete‑Continuous Generative Modeling**：

   * 先讲“混合空间的一体化生成”（CANDI）
   * 再讲“two-stage 语义规划 + 连续执行”（HDP）
   * 最后讲“long-horizon 需要上下层耦合纠错”（CHD）。

3. **Trajectory Foundation Models for Route Generation**：

   * 用 UniTraj 的“接入 ControlTraj 提升 generation 指标 + 跨城迁移”作为 foundation→generation 的标杆证据。
   * 再补 2025：GenMove（统一任务 + 条件控制，generation 提升）与 MoveGPT（跨城语义对齐 + MoE scaling）。

---

如果你希望我把这份梳理进一步变成**可直接写进论文的“Related Work”段落**（含更正式的学术表达、以及按你们论文的术语统一“route / corridor / waypoint / semantics”的定义），我也可以基于你们目标任务设定（输入输出形式、是否在路网约束下、是否 OD 条件、是否个性化偏好）把每一段落的“承上启下句”写出来。

[1]: https://arxiv.org/html/2510.14819v2 "Capturing Context-Aware Route Choice Semantics for Trajectory Representation Learning"

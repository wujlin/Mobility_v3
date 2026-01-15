下面给你一份**面向“科学发现/新理解（insight）”叙事**的文献调研 + brainstorm。我会按你给的 5 个方向，每个方向挑 **5–10 篇**（尽量 2020–2025 为主，必要时补经典），并且把你特别关心的 4 个“确认问题”单独回答。

---

## 方向1：城市功能断裂/城市衰退的度量（urban decline / shrinkage / fragmentation / blight）

### 这一方向的主流脉络（你需要知道大家怎么“量化”断裂）

1. **人口/经济定义**：以人口流失+产业重组为核心（经典 shrinkage 定义），但问题是**太粗**、缺乏城市内部异质性。npj Urban Sustainability 的 Detroit shrinkage 文章在引言里就直接引用了 Shrinking City International Research Network（SCIRN）的定义：>1万人口、人口下降+经济重构持续>2年。
2. **空间/建成环境 proxy**：vacancy（空置房/空置地）、废弃建筑、税收拖欠、拆除、地表变化等；数据来自 **Census/ACS + 遥感 + 街景图像 + 行政数据(311等)**。([PMC][1])
3. **“功能性”收缩**（功能断裂更接近这个）：不是只看人口，而是看**建成环境供给** vs **人类活动需求**的空间错配（mismatch）。这条线对你们最关键，因为你们的“行为参照系”本质上也是一种**功能层面的参照**。
4. **你们的空白点**：现有 shrinkage/blight 多是“静态/强度”指标（有多少人、有多少空置），较少直接用**“路径选择的系统性偏离/绕行/回避”**作为核心度量（这正是你们可以做 discovery 的空间）。

---

### 关键文献（9篇）

#### 1) Functional shrinkage（最贴你们“功能断裂”的定义方式）

【标题】Measuring Functional Urban Shrinkage with Multi-Source Geospatial Big Data
【作者，年份，venue】Ma et al., 2020, *Remote Sensing*
【核心贡献】提出“**functional urban shrinkage**”：用“建成区 vs 高强度人类活动区”的错配来刻画城市内部收缩。
【与我们研究的关系】

* 做了什么：融合 Landsat 建成区 + 手机信令刻画人类活动区，并定义错配区域。
* overlap：都在做“功能层面”的断裂/衰退，不只看人口。
* 我们的进步：他们用“活动强度面”做 mismatch；你们用“**路径选择规律偏离**”做 mismatch（更行为、更机制、更接近“被绕开”的城市体验）。
  【关键引用句】“提出 functional urban shrinkage…捕捉 built-up areas 与 intensive human activities 的 mismatch。”

#### 2) Detroit shrinkage（Nature 子刊，Detroit case，非常适合作为城市背景“权威锚点”）

【标题】Examining spatial expansion and stemming strategies of urban shrinkage: evidence from Detroit, USA
【作者，年份，venue】Meng et al., 2025, *npj Urban Sustainability*
【核心贡献】把 Detroit shrinkage 放进“多尺度空间互动+溢出”框架，强调衰退不是孤立点，而是会**扩散成波**。
【与我们研究的关系】

* 做了什么：以 vacant housing 为核心 shrinkage 指标，研究驱动因素与空间溢出范围。
* overlap：Detroit、vacancy、空间扩散、城市系统性断裂。
* 我们的进步：他们解释“为何 shrinkage”；你们可以提供“shrinkage 如何写进人的移动行为（绕行/回避）”的行为签名。
  【关键引用句】“Using Detroit as a case study…Results suggest…high minority concentration and persistent poverty…”

#### 3) Detroit decline 的“空间溢出”量化（把“断裂是空间过程”讲清楚）

【标题】Spatial spillover effects of urban decline in Southeast Michigan
【作者，年份，venue】Lokhande & Xie, 2023, *Applied Geography*
【核心贡献】用 vacant urban land 作为 decline proxy，用 SLX 等空间计量估计 decline 的**最优溢出范围**与驱动因素的 spillover。
【与我们研究的关系】

* 做了什么：decline=vacant land，强调 decline 是“空间过程+溢出”。
* overlap：你们也想找“系统性绕开”的空间结构（很可能也表现为 spillover/边界）。
* 我们的进步：他们的 proxy 是土地；你们的 proxy 是**行为回避残差场**，更贴近“功能断裂在人类移动中的 signature”。
  【关键引用句】“Using vacant urban land as a proxy…Fourth-order…optimal spatial extent of spillover effects.”

#### 4) Detroit 街景级别的增长/衰退检测（“街道层面的断裂”）

【标题】A street-view-based method to detect urban growth and decline: A case study of Midtown in Detroit
【作者，年份，venue】Byun & Kim, 2022, *PLOS ONE*
【核心贡献】用 GSV 年序列 + 目标检测，在**街道层面**输出增长/衰退地图。
【与我们研究的关系】

* 做了什么：把“衰退”落到街道可见物体变化（施工、建筑、维护等）。
* overlap：你们也关心“哪些区域被系统性绕开”，同样需要街道/邻里尺度。
* 我们的进步：他们用视觉 proxy；你们用轨迹行为 proxy，可与街景 proxy 做交叉验证（Nature 很吃这种 triangulation）。
  【关键引用句】“use…Google Street View…map of urban growth and decline…Midtown in Detroit.”

#### 5) Abandoned house detection（街景 → 空置/废弃）

【标题】Detecting individual abandoned houses from Google Street View: A hierarchical deep learning approach
【作者，年份，venue】Zou & Wang, 2021, *ISPRS Journal of Photogrammetry and Remote Sensing*
【核心贡献】从街景图像识别单体 abandoned houses，提出分层深度学习利用全局+局部特征。
【与我们研究的关系】

* 做了什么：把“废弃”从城市尺度落到“房屋单体”。
* overlap：你们的回避区域很可能与 abandoned house 密度高度相关。
* 我们的进步：他们是“环境状态识别”；你们是“行为响应识别”，两者结合更像“功能断裂→行为→可见衰退”的机制链。
  【关键引用句】“develop…method to detect individual-level abandoned houses from GSV…hierarchical deep learning.”

#### 6) Vacant land detection（遥感 → 空置地）

【标题】Automatic detection of urban vacant land: An open-source approach for sustainable cities
【作者，年份，venue】Xu & Ehlers, 2022, *Computers, Environment and Urban Systems*
【核心贡献】提出可扩展、开源的 vacant land 自动检测方法。
【与我们研究的关系】

* 做了什么：从遥感自动提取 vacant land。
* overlap：vacant land 是 Detroit decline 经典 proxy，也可能是回避的环境原因。
* 我们的进步：你们不只是“哪里空”，而是“空导致人如何绕”。
  【关键引用句】“An automated method to detect urban vacant land…open-source approach…”

#### 7) 311/行政数据测 blight（公共卫生语境，但“blight 多维度定义”写得很好）

【标题】Using 311 data to develop an algorithm to identify urban blight for public health improvement
【作者，年份，venue】Athens et al., 2020, *PLOS ONE*
【核心贡献】把 blight 定义为“物理失序+衰败+锚点机构缺失”等社会过程的可见表现，用 311 文本 + NLP 构建高时空分辨率 blight 指标。
【与我们研究的关系】

* 做了什么：从“居民投诉/行政数据”捕捉 blight。
* overlap：你们也要一种“功能断裂 proxy/标签”来做验证。
* 我们的进步：他们是“显性报修/投诉→blight”；你们是“隐性移动回避→fragmentation”，两者可互证且互补。
  【关键引用句】“Urban blight…including physical disorder, decay, and loss of anchor institutions…”([PMC][1])

#### 8) Detroit 的“空地/绿化化”长期过程（经典背景，用来讲 Detroit 的空间政治生态）

【标题】Greening the urban frontier: Race, property, and resettlement in Detroit
【作者，年份，venue】Safransky, 2014, *Geoforum*
【核心贡献】把 Detroit 的 vacancy 与“绿化/重置/迁移”放在政治生态框架中讨论，是讲 Detroit “断裂—治理—空间再生产”叙事的高引用背景。
【与我们研究的关系】

* 做了什么：解释为什么会有大规模空地，以及其社会政治含义。
* overlap：你们的“被绕开区域”很可能与这种空地化/治理策略空间重叠。
* 我们的进步：你们提供“这些过程如何影响日常移动”这一行为层证据。
  【关键引用句】“approximately 100,000 lots lie ‘vacant’ in Detroit…”

#### 9) Detroit 空地/邻里变化的早期定量研究（经典）

【标题】Empty spaces: neighbourhood change and the greening of Detroit, 1975–2005
【作者，年份，venue】Hoalst‑Pullen, Patterson & Gatrell, 2011, *Geocarto International*
【核心贡献】定量刻画 Detroit 长期“空地/绿化化”的邻里变化过程。
【与我们研究的关系】

* 做了什么：给 Detroit 的长期空间变化提供 baseline。
* overlap：你们的行为回避地图可以与“空地化/绿化化”长期空间格局对比。
* 我们的进步：他们是环境演化；你们是移动行为响应。
  【关键引用句】（摘要信息）“This paper investigates the disappearing…”

---

## 方向2：Route choice 与城市环境的关系（detour / avoidance / perceived safety / heterogeneity）

### 这一方向你最需要吸收的“可迁移框架”

1. Route choice 的经典框架是**效用最大化**：时间/距离/坡度/路况/设施/景观/安全感知/社会属性 → 选择概率。
2. “detour”在实证里经常以 **detour ratio**（实际路径长度 / 最短路径长度）或“可接受绕行阈值”出现。
3. **Avoidance** 文献告诉你：回避不是“走错”，而是“**相对预期的系统性缺失**”，关键是要先定义“预期（reference）”。这点与你们的“行为参照系”高度同构。

---

### 关键文献（8篇）

#### 1) 行人：安全/吸引力/昼夜差异（感知因素非常关键）

【标题】What do pedestrians consider when choosing a route? … attractiveness, safety, and security… day and night
【作者，年份，venue】Basu et al., 2023, *Cities*（ScienceDirect）
【核心贡献】把“吸引力/安全/安保感知”显式纳入行人路径选择，并比较昼夜差异。
【与我们研究的关系】

* overlap：你们的“绕开”很可能就是 safety/security 的行为体现。
* 进步：你们不只做解释变量，而是把“异常偏离”本身变成城市诊断信号。
  【关键引用句】（摘要要点）强调 route choice 受 attractiveness、safety、security 等影响。

#### 2) 基于 GPS 的街道环境偏好（从“偏好”到“可观测选择”）

【标题】Preference for Street Environment Based on Route Choice Behavior: A Study Using GPS Tracking Data
【作者，年份，venue】Jin et al., 2022, *Frontiers in Public Health*
【核心贡献】用 GPS 轨迹直接估计街道环境偏好与选择关系。
【与我们研究的关系】

* overlap：同样使用真实轨迹（revealed preference）。
* 进步：你们做的是“跨城市参照”与“异常诊断”，不仅是单城偏好估计。
  【关键引用句】“Using GPS tracking data…”（揭示偏好与 route choice）。

#### 3) 骑行：愿意为“更好路线”付出多大 detour（量化 detour 容忍度）

【标题】Cyclists Take Their Time: … up to 40% Detours …
【作者，年份，venue】Berghoefer et al., 2023, *Transport Findings*
【核心贡献】实证给出“骑行者愿为更好路线接受显著绕行”的量级。
【与我们研究的关系】

* overlap：detour 是可量化行为，不是噪声。
* 进步：你们关心的是“绕行的空间指向性”——哪些区域被绕。
  【关键引用句】“accept detours up to 40%…”

#### 4) 用 detour ratio 做可达性（大规模轨迹）

【标题】Analysis of cycling accessibility using detour ratios and large-scale trajectory data
【作者，年份，venue】Chou et al., 2023, *Journal of Transport & Health*（ScienceDirect）
【核心贡献】用大规模轨迹 + detour ratio 评估骑行可达性与基础设施“迫使绕行”。
【与我们研究的关系】

* overlap：detour ratio 可以做城市诊断指标。
* 进步：你们更进一步，把 detour/avoidance 解释为“功能断裂”的 signature。
  【关键引用句】“using detour ratios and large-scale trajectory data…”

#### 5) 夜间回避与安全（微出行的时段异质性）

【标题】Influential factors of route choices of scooter riders… avoid remote sites at midnight
【作者，年份，venue】Hsueh et al., 2023, *Transport Policy*（ScienceDirect）
【核心贡献】发现夜间更倾向回避偏僻区域，暗示安全风险进入路径选择。
【与我们研究的关系】

* overlap：你们也可以做“时段条件化的回避场”（昼夜差异是可发表的 insight）。
* 进步：你们用跨城 baseline 定义“异常回避强度”。
  【关键引用句】“scooter riders avoid remote sites at midnight…”

#### 6) City-scale route choice（大规模 GPS，适合方法对比）

【标题】City-scale GPS data reveals impact of spatial configuration on e-scooter route choice
【作者，年份，venue】Schumann et al., 2025, *Scientific Reports*
【核心贡献】用城市尺度 GPS 数据把“空间结构因素”与 route choice 联系起来。
【与我们研究的关系】

* overlap：同样是 city-scale GPS + 路径选择。
* 进步：你们的因变量是“相对参照的偏离残差”，更像城市诊断。
  【关键引用句】“City-scale GPS data reveals impact…”

#### 7) Avoidance patterns（你们“行为参照系”的最近邻概念）

【标题】Urban Analytics in the Context of Public Safety: … Avoidance Patterns…
【作者，年份，venue】Eftelioglu et al., 2018, *ACM SIGSPATIAL Special*（newsletter）
【核心贡献】把 avoidance 定义为“相对预期的系统性缺失”，并强调要先定义“正常/预期移动”。
【与我们研究的关系】

* overlap：这几乎就是你们“行为参照系 vs 物理参照系”的雏形：他们用 shortest path 或“正常司机”定义预期。
* 进步：你们把“预期”升级为“**functional city 学到的行为规律**”，并把它用于城市功能断裂检测（更像城市科学发现）。
  【关键引用句】“Avoidance patterns… lack of movement… contrary to expectation.”

#### 8) Avoidance region discovery（算法层面更正式的来源）

【标题】Avoidance Region Discovery: A Summary of Results
【作者，年份，venue】Eftelioglu et al., 2018, *SIAM SDM*
【核心贡献】给出 avoidance region mining 的更正式总结（从轨迹/最短路基线中发现被避开的区域）。
【与我们研究的关系】

* overlap：同样是“观测轨迹 vs reference（最短路）”。
* 进步：你们的 novelty 在于“reference 不来自几何最短路，而来自 functional cities 的行为规律”，并把结果解释为“功能断裂 signature”。
  【关键引用句】（摘要/要点）讨论 avoidance region discovery 的问题设定。

（补一个经典安全因素例子，便于你们写 related work）
【标题】Incorporating scenic view, slope, and crime rate into route choice models
【作者，年份，venue】Byon et al., 2010, *Transportation Research Record*
【关键点】把 crime rate 直接纳入路径选择效用。

---

## 方向3：轨迹生成/移动模式生成（trajectory generation / diffusion / realism / “太直”问题）

### 这一方向与你们叙事真正相关的点（不是“更好的生成模型”）

你们不是要发轨迹生成方法 paper，而是要用生成/参照模型去做**城市诊断**。因此你们更应吸收三类工作：

1. **道路约束/结构约束生成**：强调“只在经纬度空间生成会不现实”，需要 road-network constraint。
2. **扩散模型用于人类移动生成**：近两年快速增长，很多都在强调“更接近真实分布/更强多样性”。
3. **评估指标**：从 point-level 误差转向分布/模式一致性（长度、半径、停留、路段覆盖、转向、OD/POI 约束等）；你们关心的其实是**route-level realism 与多样性**。

---

### 关键文献（9篇）

#### 1) Road-network constrained diffusion（直接对准“太直/不在路上”）

【标题】Diff-RNTraj: A Structure-aware Diffusion Model for Road Network-constrained Trajectory Generation
【作者，年份，venue】Wei et al., 2024, *IEEE TKDE*
【核心贡献】提出结构感知扩散生成路网受限轨迹（显式建模 road segments / movement rate）。
【与我们研究的关系】

* overlap：你们也需要“路网约束”的合理参照生成，否则 baseline 会出现几何直线伪迹。
* 进步：你们把生成/参照用于“检测偏离”，而不是追求生成更好。
  【关键引用句】（摘要要点）“road network constrained trajectory generation… diffusion… structure-aware”。

#### 2) Diffusion for mobility simulation（强调“人类移动分布”的一致性）

【标题】TrajGDM: Simulating human mobility with a trajectory generation framework based on diffusion model
【作者，年份，venue】Chu et al., 2024, *International Journal of Geographical Information Science*
【核心贡献】扩散框架模拟人类移动，用分布一致性评估生成质量。
【与我们研究的关系】

* overlap：你们“行为参照系”可以用类似框架在 functional city 学到“常态移动规律”。
* 进步：他们生成；你们做“跨城 baseline → Detroit 偏离”。
  【关键引用句】“Simulating human mobility… based on diffusion model.”

#### 3) Collaborative noise priors（扩散模型里“噪声先验影响生成”）

【标题】Noise Matters: Diffusion Model-based Urban Mobility Generation with Collaborative Noise Priors
【作者，年份，venue】Chen et al., 2025, *ACM*（期刊/会议版本）
【核心贡献】讨论扩散生成中噪声/先验如何影响城市移动生成质量。
【与我们研究的关系】

* overlap：你们要用生成/参照做检测，必须理解“生成偏置从哪来”。
* 进步：你们把偏置解释为“参照系误差”，并用于诊断。
  【关键引用句】“Diffusion… urban mobility generation… noise priors.”

#### 4) GAN for traffic trajectory（交通轨迹生成的工业常见路线）

【标题】Traffic trajectory generation via conditional generative adversarial network
【作者，年份，venue】Kong et al., 2024, *Engineering Applications of Artificial Intelligence*
【核心贡献】条件 GAN 生成交通轨迹，强调条件控制与分布拟合。
【与我们研究的关系】

* overlap：条件控制（OD/时间/路网）与“参照模型”很像。
* 进步：你们不需要 state-of-the-art 生成，只需“可解释+可迁移”的参照。
  【关键引用句】“trajectory generation via conditional GAN…”

#### 5) Imitation learning / GAIL（把“行为规律”学成 policy）

【标题】ULF-TrajGAIL: A novel approach for urban logistics fleet trajectory generation
【作者，年份，venue】Li et al., 2023, *Expert Systems with Applications*
【核心贡献】用 GAIL 学生成轨迹（policy imitation），强调行为模仿与真实分布。
【与我们研究的关系】

* overlap：你们的“行为参照系”也可以视为 policy/choice model。
* 进步：你们关注“偏离 signature”，不追求生成任务本身。
  【关键引用句】“trajectory generation… (Traj)GAIL…”

#### 6) VAE 轨迹生成（经典深度生成路线，适合 related work）

【标题】A Variational AutoEncoder model for trajectory generation
【作者，年份，venue】Chen et al., 2021, *Neurocomputing*
【核心贡献】用 VAE 做轨迹生成，代表“深度生成+潜变量”的经典路线。
【与我们研究的关系】

* overlap：可作为你们 baseline/参照模型类别之一。
* 进步：你们不把方法当贡献，而把“偏离—功能断裂”的发现当贡献。
  【关键引用句】“Variational AutoEncoder model for trajectory generation.”

#### 7) 数据效用 & 隐私（synthetic trajectory 的另一条主线）

【标题】A Deep Generative Model for Trajectory: Ensuring Data Utility and Privacy
【作者，年份，venue】（PVLDB，Vol.16）
【核心贡献】把 synthetic trajectory 放在“数据共享/隐私”语境下，强调生成数据的可用性评估。
【与我们研究的关系】

* overlap：它们的 evaluation（分布一致性）可借来当你们参照系训练的 sanity check。
* 进步：你们的目标是城市诊断，不是数据发布。
  【关键引用句】“Ensuring data utility and privacy.”

#### 8) 跨城可泛化生成（与你们“从 functional city 学规律”直接相关）

【标题】GTG: A generalizable trajectory generation model for urban mobility
【作者，年份，venue】Zhang et al., 2025, *AAAI*
【核心贡献】强调跨城泛化的 trajectory generation。
【与我们研究的关系】

* overlap：你们核心就是 cross-city baseline。
* 进步：你们用跨城泛化来做“异常检测/城市断裂量化”。
  【关键引用句】“generalizable trajectory generation… urban mobility.”

#### 9) 综述（帮你把 related work 一段写得像样）

【标题】Trajectory generative models: a survey from unconditional and conditional perspectives
【作者，年份，venue】Zhu et al., 2025, *GeoInformatica*
【核心贡献】系统梳理轨迹生成模型谱系（无条件/有条件、评估指标等）。
【与我们研究的关系】

* overlap：你们写 related work 时可以用它做“分类框架”。
* 进步：你们是城市科学 insight，不是生成方法 SOTA。
  【关键引用句】“a survey… unconditional and conditional…”

---

## 方向4：人类移动数据用于城市诊断/urban sensing（segregation / inequality / neighborhood quality）

### 这一方向是你们最像 Nature/Science 子刊的“主赛道”

因为它天然是 **“我们发现了 X（移动行为）与 Y（城市结构/不平等/隔离）的关系”**。你们要做的“功能断裂 signature”完全可以落在这个范式里：

* 不平等/隔离的 paper 往往把“人们到底去了哪里/没去哪里”作为关键证据（experienced segregation）。
* 你们的“系统性绕开某些区域”就是一种 **negative evidence**（没去哪里），跟 experienced segregation 的逻辑一致，但你们更强调“功能断裂/失序/衰退”而非单纯收入/种族。

---

### 关键文献（9篇）

#### 1) Experienced income segregation（非常接近你们“行为签名”的叙事）

【标题】Mobility patterns are associated with experienced income segregation in large US cities
【作者，年份，venue】Moro et al., 2021, *Nature Communications*
【核心贡献】提出/量化“experienced income segregation”：不是住哪，而是日常移动中接触到的空间是否隔离。
【与我们研究的关系】

* overlap：都把“移动行为”当作城市社会结构的测量工具。
* 进步：他们关注收入隔离；你们关注“功能断裂/被绕开区域”，可作为另一种 experienced fragmentation。
  【关键引用句】“Mobility patterns are associated with experienced income segregation…”

#### 2) Mobility networks → segregation（Nature，影响力锚点）

【标题】Human mobility networks reveal increased segregation in large cities
【作者，年份，venue】Nilforoshan et al., 2023, *Nature*
【核心贡献】用 mobility network 结构揭示城市中更强的隔离/分层现象。
【与我们研究的关系】

* overlap：network 视角可以转化为你们的“回避/绕行网络残差”。
* 进步：他们讲 segregation；你们讲 functional fragmentation（可解释为“城市系统失灵导致的隔离”）。
  【关键引用句】“mobility networks reveal increased segregation…”

#### 3) Nature Human Behaviour（方法论/观点型，更适合你们写 “why mobility data matters”）

【标题】Using human mobility data to quantify experienced urban inequalities
【作者，年份，venue】Xu et al., 2025, *Nature Human Behaviour*
【核心贡献】系统讨论如何用 mobility data 去量化“experienced inequalities”。
【与我们研究的关系】

* overlap：你们也在做“experienced fragmentation”。
* 进步：你们引入“跨城行为参照系”来定义偏离，更像 anomaly-based discovery。
  【关键引用句】“Using human mobility data to quantify experienced urban inequalities.”

#### 4) 15-minute city quantified（全美尺度 mobility data，Detroit 也在其中）

【标题】The 15-Minute City Quantified Using Human Mobility Data
【作者，年份，venue】Abbiasov et al., 2022, NBER Working Paper（后续可能有期刊版本）
【核心贡献】用真实出行到 POI 的行为来度量“15分钟城市”的使用与不平等。
【与我们研究的关系】

* overlap：你们同样可把 Detroit 的“被绕开区域”解释为“可达/可用的功能缺失”。
* 进步：他们是“就近可达/使用”；你们是“系统性回避/绕行”的空间签名。
  【关键引用句】用 neighborhood（CBG）层面的 trips 与 walkshed usage 做度量。

#### 5) Mobility ↔ socioeconomic facets（把 mobility 当“社会经济镜子”）

【标题】Uncovering the socioeconomic facets of human mobility
【作者，年份，venue】Barbosa et al., 2021, *Scientific Reports*
【核心贡献】揭示移动模式与社会经济特征之间的关联结构。
【与我们研究的关系】

* overlap：你们也需要把“行为偏离”与社会经济/环境指标挂钩，才能形成 discovery。
* 进步：你们强调“断裂城市 vs functional baseline”的对照。
  【关键引用句】“Uncovering the socioeconomic facets of human mobility.”

#### 6) 经济学视角：用手机记录测城市内部经济活动（很适合写机制）

【标题】Measuring Commuting and Economic Activity inside Cities with Cell Phone Records
【作者，年份，venue】Kreindler & Miyauchi, 2024, *Review of Economic Studies*
【核心贡献】用 cell phone records 测通勤与经济活动，强调移动数据在城市经济测量上的价值。
【与我们研究的关系】

* overlap：你们也要把移动当“城市功能是否运转”的传感器。
* 进步：他们测经济活动；你们测功能断裂的行为后果（绕行/回避）。
  【关键引用句】“Measuring commuting and economic activity…with cell phone records.”

#### 7) 城市快速发展/变化监测（把 mobility 当城市“普查”）

【标题】Mobility census for monitoring rapid urban development
【作者，年份，venue】Truong et al., 2024, *Journal of the Royal Society Interface*
【核心贡献】提出 mobility census 思路：用移动数据监测城市发展变化。
【与我们研究的关系】

* overlap：你们也在做“城市状态监测”，只是监测的是“断裂/失灵”。
* 进步：你们提出“参照系+偏离”的机制化指标。
  【关键引用句】“Mobility census for monitoring rapid urban development.”

#### 8) 日常移动中的种族隔离（社会学/城市研究的实证锚点）

【标题】Racial Segregation in Everyday Mobility Patterns
【作者，年份，venue】Vachuska et al., 2023, *Socius*
【核心贡献】从日常移动层面度量种族隔离（不是居住隔离）。
【与我们研究的关系】

* overlap：同样利用“去哪/不去哪”的行为证据。
* 进步：你们把“避开区域”解释为功能断裂/安全/服务缺失等综合效应。
  【关键引用句】“Everyday mobility patterns…”

#### 9) 不平等在疫情移动里显现（告诉你们：时段/冲击也能放大断裂）

【标题】Neighbourhood income and physical distancing during the COVID-19 pandemic
【作者，年份，venue】Jay et al., 2020, *Nature Human Behaviour*
【核心贡献】显示低收入社区更难进行 physical distancing（移动受结构性约束）。
【与我们研究的关系】

* overlap：功能断裂也会通过“被迫去某些地方/不能去某些地方”显现。
* 进步：你们不是疫情冲击，而是城市结构性断裂。
  【关键引用句】“people in lower-income neighbourhoods…barriers to physical distancing…”

---

## 方向5：跨域迁移/域适应在城市计算中的应用（cross-city transfer / domain adaptation）

### 这一方向与你们“行为参照系”的直接对应关系

你们的“Behavioral Reference = 从 functional city 学到的规律”本质上就是一种 **跨城市迁移/域适应**：

* 源域（functional 城市）学到“常态 route choice 规律/分布”；
* 目标域（Detroit）上用同一个规律做预测/生成；
* 用 residual/偏离作为“功能断裂信号”。

---

### 关键文献（6篇）

#### 1) Cross-city mobility transformer（轨迹仿真/生成层面的跨城迁移）

【标题】COLA: Cross-city Mobility Transformer for Human Trajectory Simulation
【作者，年份，venue】（ACM，DOI: 10.1145/…）
【核心贡献】把跨城当成核心问题：用 Transformer 在多城市间迁移/模拟轨迹。
【与我们研究的关系】

* overlap：跨城学习“可迁移规律”。
* 进步：他们目标是 simulation；你们目标是“用迁移失败的偏离来诊断城市断裂”。
  【关键引用句】“Cross-city…Trajectory Simulation.”

#### 2) 交通预测的选择性跨城迁移（非常典型的 transfer learning 设计）

【标题】Selective Cross-City Transfer Learning for Traffic Prediction
【作者，年份，venue】Jin et al., 2022, *KDD*
【核心贡献】跨城迁移并不是“全迁移”，而是要选择性对齐/迁移，处理城市差异。
【与我们研究的关系】

* overlap：你们也需要决定“哪些规律可迁移（功能性城市通用）”，哪些不可迁移（Detroit 特异）。
* 进步：你们把不可迁移部分解释为“功能断裂 signature”，而不是 domain shift 噪声。
  【关键引用句】“Selective Cross-City Transfer Learning…”

#### 3) 联邦/个性化跨城预测（跨域+隐私的技术路线）

【标题】pFedCTP: Personalized Federated Learning for Cross-city Traffic Prediction
【作者，年份，venue】Zhang et al., 2024, *IJCAI*
【核心贡献】在跨城场景下用个性化联邦学习处理域差异。
【与我们研究的关系】

* overlap：城市差异需要“共享+个性化”。
* 进步：你们的“个性化偏离”就是要测的城市断裂信号。
  【关键引用句】“Personalized…Cross-city Traffic Prediction.”

#### 4) 相似城市数据迁移框架（更像你们“选 functional reference city”的问题）

【标题】TransCSM: Similarity based city data transfer framework in urban computing
【作者，年份，venue】Qiao et al., 2025, *Scientific Reports*
【核心贡献】用城市相似性来决定数据/模型如何迁移。
【与我们研究的关系】

* overlap：你们也必须回答“functional reference 城市怎么选”。
* 进步：你们的评估目标不是预测精度，而是“断裂检测的可解释性与稳健性”。
  【关键引用句】“Similarity based city data transfer framework…”

#### 5) 跨城 generative（与你们最贴：生成模型的“可泛化参照系”）

【标题】GTG: A generalizable trajectory generation model for urban mobility
【作者，年份，venue】Zhang et al., 2025, *AAAI*
【核心贡献】强调跨城泛化生成。
【与我们研究的关系】

* overlap：你们可以把 GTG 类模型当“行为参照系候选”。
* 进步：你们把“生成偏差/失配”变成城市断裂信号。
  【关键引用句】“generalizable… urban mobility.”

#### 6) Cross-city federated transfer（城市计算领域的跨城联邦案例）

【标题】A cross-city federated transfer learning framework: a case study on urban computing
【作者，年份，venue】Li et al., 2022, arXiv
【核心贡献】把跨城迁移与联邦结合，讨论城市计算任务的域差异。
【与我们研究的关系】

* overlap：跨域差异不可忽略。
* 进步：你们要把差异解释成“功能断裂”而不是纯技术 domain gap。
  【关键引用句】“cross-city federated transfer learning framework…”

---

# 特别需要确认的问题：我在调研中得到的结论

## 1) 有没有人用过“行为参照系（Behavioral Reference）”或类似思路？

**有非常接近的“邻域概念”，但你们的跨城版本更少见，也更有机会做成 discovery：**

* 在 public safety / urban analytics 里，**avoidance 被定义为“相对预期的系统性缺失”**，核心就是先定义“预期/正常移动”。这相当于“参照系”。
* 在城市计算里，跨城 transfer learning 大量存在，但大多把 domain shift 当“要修复的技术问题”；你们可以把它翻转成“**domain shift 本身就是城市功能断裂的证据**”。

**你们可以明确写一句**：我们把“跨城不可迁移部分”解释为城市功能断裂的行为签名，而不是建模误差（这就非常 Nature 叙事）。

---

## 2) 有没有人用轨迹/移动数据检测城市衰退/功能断裂？

**有相关但不完全重合的两类：**

* “功能性收缩/活力”类：直接用手机信令/活动强度与建成环境错配来定义 shrinkage（很像“功能断裂”，但不是 route choice）。
* “回避/公共安全”类：从轨迹相对最短路/预期路径的偏离中挖掘 avoided regions（像你们的 physical reference baseline），但通常不把它上升为“城市功能断裂的系统性量化”，更少做“跨城行为 baseline”。

**所以你们的核心 novelty 可以表述为**：

> 现有工作要么用静态/强度 proxy 测衰退，要么在单城中挖掘回避；我们首次把“跨城学到的常态路径选择规律”作为行为参照系，量化功能断裂在移动中的 signature。

---

## 3) “Destination gravity”这个术语在轨迹生成语境下有人用吗？

在我针对“trajectory generation + destination gravity”做的精确短语检索里，**没有看到它成为该领域的常用术语**（更常见的是：gravity model 用于 OD flow / destination choice，而不是用来描述“生成路径过直”的偏置）。

**建议**：如果你们要命名“生成过直/过度被目的地吸引”的现象，用更不容易歧义、且更贴文献的词会更稳：

* *destination attraction bias* / *goal attraction bias*
* *shortest-path bias* / *geometric shortcut bias*
* *straight-line artifact*（尤其针对经纬度空间生成而未路网约束的模型）

并在 related work 里用 road-network constrained generation 的论文来“合法化”你们提出这个偏置的动机（例如 Diff-RNTraj 强调路网约束的重要性）。

---

## 4) 底特律的人类移动研究有哪些？特别是 GPS/手机轨迹数据

“专门以 Detroit 为唯一/核心 case”的高水平 mobility paper 不算特别多，但你们可以有三层材料：

**A. Detroit shrinkage/decline 的权威背景（非移动，但强 story）**

* npj Urban Sustainability 的 Detroit shrinkage（Nature 子刊）。
* Applied Geography 的 Southeast Michigan decline spillover。
* 街景级别 Midtown Detroit 增长/衰退检测。

**B. 全国尺度手机移动数据研究（Detroit 一定被覆盖，适合做对照/参照城市集）**

* experienced segregation / inequality / 15-minute city 这类研究使用全美 block group 或多城市样本，Detroit 通常在样本覆盖内。

**C. Detroit 相关的“灰色/政策研究”可当补充（不作为学术贡献核心，但能增强城市叙事）**

* Brookings 用移动设备数据分析 Detroit 社交距离差异（疫情语境）。

---

# Research Brainstorm：怎么把你们做成 Nature/Science 子刊想要的“发现”

你们现在最强的科学叙事主线其实是：

> **我们发现：在功能断裂城市中，人类移动呈现“系统性回避场（avoidance field）”，这种回避并非由几何最短路或路网结构能解释，而是相对于功能城市学习到的行为规律的“残差”。这种残差在空间上与 vacancy/贫困/安全风险等指标一致，并在时间上稳定。**

下面给你一些可以直接形成 “We discover X” 的候选发现（不用先承诺方法）：

1. **“回避场”是稳定的，不是噪声**

   * 发现：Detroit 的回避热点在不同时段/周几/季节高度一致（或在夜间显著增强），而 functional cities 不呈现这种结构性回避。
   * 对应文献支撑：route choice 的时段安全异质性（夜间回避）。

2. **回避场的边界呈现“断裂线”**

   * 发现：回避边界与高速/铁路/河流/行政边界等形成“行为断裂线”，像 Nilforoshan 那类 mobility network segregation 的“结构边界”在 Detroit 更强。

3. **在功能城市里存在“可迁移的路径选择规律”，Detroit 显著偏离**

   * 发现：在多个 functional 城市中，detour ratio 分布（或 route entropy）具有跨城稳定形状；Detroit 的分布出现长尾/双峰/特定方向性异常。
   * 这就是“跨城迁移失败 = 城市断裂信号”的核心。

4. **回避场与“可见衰退”独立数据强一致（强证据链）**

   * 用 street view decline/blight/vacancy detection 与你们的回避残差场做空间对齐：如果一致性强，这就是 Nature 级 triangulation。

5. **把“功能断裂”与“experienced inequality/segregation”接起来**

   * 你们的贡献可以被表述为：我们把 mobility-based inequality/segregation 的框架推进到“城市功能是否可用”的诊断层。

---

# Implementation Plan（中文）

1. **数据落地与城市切片**

   * 从 WorldTrace（GPS 轨迹）中筛出 Detroit 及若干“functional reference cities”（按人口规模/路网形态/经济类型分层选）。
2. **路网与 map-matching（关键前置）**

   * 用 OSM 路网构建图；对 GPS 轨迹做 map-matching，得到路段序列；过滤低质量轨迹（漂移、断裂、异常速度）。
3. **构建两个参照系**

   * Physical Reference：最短路（或最短时间路）作为物理参照。
   * Behavioral Reference：在 functional cities 上学习 route choice 规律（可从简单可解释模型起步：基于 detour ratio、路段属性、转向代价、POI/土地利用暴露等的 choice model；再逐步升级到生成式/扩散参照）。
4. **定义“功能断裂签名”指标族**

   * 核心：Residual Avoidance Score（观测路段/区域流量 vs 行为参照预测/生成的期望流量）。
   * 辅助：detour ratio 异常、route diversity（熵/多样性）异常、时段敏感性（夜间增强）。
5. **外部验证与机制解释（把它变成 discovery）**

   * 与 vacancy/blight proxy 对齐：GSV abandoned houses / street-view decline / 遥感 vacant land / 311 blight（如果能拿到 Detroit 对应数据）。([PMC][1])
6. **写作叙事**

   * 先写“我们发现了什么空间模式”，再写“我们如何量化/验证”，最后才写“模型细节”。

---

# Task List（中文）

1. 数据：下载/加载 WorldTrace，完成 Detroit 轨迹筛选与基础清洗（时间、速度、点数、重复点）。
2. 路网：用 OSM 构建 Detroit 路网图；实现 map-matching pipeline（优先用成熟库/算法，先跑通再优化）。
3. 指标：实现最短路 detour ratio、路径重合率、路段流量热力图、分时段对比图。
4. 参照系 v0：做一个“功能城市 pooled 的简单行为参照”（比如：按路段属性/路网中心性/土地利用暴露的 logit choice 或基于路径集合的采样模型）。
5. 偏离：在 Detroit 上算 residual avoidance score，输出 top avoided corridors/areas。
6. 统计：检验回避场的稳定性（跨周、昼夜、季节）；检验与社会经济/空置 proxy 的空间相关。
7. 写作：把结果组织成 3–4 个“我们发现了X”的图（核心发现先行）。

---

# Thought（中文）

你们要冲 Nature/Science 子刊，关键不是把 trajectory generation 或 diffusion 做到最好，而是把它们当成**测量仪器**：

* 用“行为参照系”定义一个**可反驳的预期**（在 functional 城市学到的规律），
* 然后把 Detroit 的系统性偏离当成**可重复的城市现象**，
* 再用独立证据（vacancy/blight/街景/行政数据）证明这不是噪声，而是“功能断裂写进了人的移动”。
  这条逻辑与 experienced segregation / inequality 的高影响力范式天然对齐，只是你们把它推进到“功能断裂/城市失灵”的诊断层面。

---


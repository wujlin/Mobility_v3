# Phase D 路线图（OSM 可行域 + 拓扑 + 城市语义 + Diffusion 多模态）

> **这是一次方向转折点**：我们从“在弱地图上用软约束逼模型学可行域”转向“补足必要信息（可行域/拓扑/语义），再让生成模型负责多模态”。  
> **核心原则**：先修输入信息与评估口径，再谈模型表达；每个阶段必须有 **可证伪审计**，禁止无意义烧卡。

---

## 0. 基于最新审计的精确诊断（为什么必须转向）

> 这里不讲符号，只讲“指标在说什么”。

### 发现 1：Macro 决策偏直（under-detour），这是 G2 主矛盾

- 证据：`detour_scalar_direction_audit`
  - GT：`max_dev_ratio p50≈0.495`，`len_ratio p50≈1.630`
  - 当前模型（MacroSkel）：明显更小（更直更短）
  - DetRes 能拉回一截，但无法凭空创造绕路拓扑

**解释**：Macro 目前的“路线选择”本质在回归直线/最短路径附近，缺少“为什么要绕”“绕到哪条走廊”的信息。

### 发现 2：mask 孔洞制造“假切墙”，导致 CUT 指标被污染

- 证据：`oracle_cut_cause_audit`
  - strict `count>=1` 下 oracle 也 CUT
  - 轻微 dilation（1 cell）即可消掉绝大多数 CUT（>90%）

**解释**：这不是模型必须背的锅，而是 weak-map 可行域代理（count）过碎/有孔洞。

### 发现 3：end 选错走廊（corridor error）占比高于“距离不对”

- 证据：`end_imprecision_audit`
  - `corridor_error` 明显高于 `dist_error`

**解释**：模型“朝着目的地走”的方向感是有的，但在目的地附近**选错平行道路/走廊**，本质是缺少道路拓扑信息（连通性与走廊区分）。

---

## 1. 新框架总览（AR 必须保留）

### 1.1 组件分工（要把责任边界钉死）

- **OSM 可行域（hard feasibility）**：回答“能不能走”（合法性/支持集），用于替换 count-based mask，消除假切墙。
- **道路拓扑（topology）**：回答“哪条走廊能到、哪条更近”（corridor selection）。
- **城市语义（POI/功能区/建成环境）**：回答“为什么要绕、绕到哪里更像真的”（绕路动机/语义一致性）。
- **AR 解码（结构机制）**：回答“怎么选才能连贯”（wp2 依赖 wp1，end 依赖 wp1+wp2；连通性写回模型）。
- **Diffusion（多模态）**：回答“有哪些不同路线风格”（前提是单条路线质量已足够）。
- **Micro DetRes（执行层）**：回答“怎么走得像真的”（局部动力学与形状质感）；不负责宏观绕路拓扑。

一句话：**AR 是“怎么选”的机制；拓扑/语义是“选什么”的信息；Diffusion 是“多样性”的来源。**

---

## 1.2 需要立刻拍板的“数据/口径决策”（reviewer checklist）

### A) 轨迹数据时间范围（决定 OSM/POI 的时间一致性口径）

- **轨迹时间范围（已知）**：`2011-04-18 ~ 2011-04-26`（深圳出租车 GPS）
- **影响**：
  - OSM/POI 若使用 2024 版本，存在“城市形态变化”的风险（尤其 POI 更严重）。
  - Stage 1/2 的目标是“道路连通/走廊选择”，道路网络相对更稳定；Stage 3 的语义结论需要更谨慎（必须做可证伪 ablation）。

### B) OSM 道路类型筛选（Stage 1 必须明确）

**默认纳入（车辆可行）**：
- `motorway, trunk, primary, secondary, tertiary, residential`

**需要拍板（reviewer 问题）**：
- `service`：**建议纳入（YES）**（停车场/加油站/内部道路对出租车可能真实可达；且能显著减少“孔洞/断裂”）
- `unclassified`：**建议纳入（YES）**（OSM 中很多城市支路会落在该类；排除会制造断裂）

**明确排除（非机动车/步行）**：
- `footway, cycleway, path, steps`（以及类似的 pedestrian-only 类别）

> 注：OSM 是中心线，落到 110m/格 的栅格时必须做“道路宽度 buffer”，否则即使 road type 正确也会出现孔洞。

### C) POI 主数据源（Stage 3）

- **建议主源**：高德（Amap）  
  理由：分类体系更标准、商业/出行类 POI 覆盖更贴近出租车行为。
- **腾讯 POI**：作为可选补充（后续统一映射到一级粗分类后再融合）。

### D) 坐标系（Stage 1 的成败关键）

OSM 默认是 WGS84；而你们原始轨迹经纬度**尚未用外部地图对齐验证**，可能出现 WGS84/GCJ-02 偏移。

**不拍脑袋结论**：Stage 1 增加一个“对齐审计”来判定坐标系：
- 把轨迹点投到 OSM mask 后统计 “on-road ratio”
- 如 on-road ratio 极低（例如 <30%），优先怀疑坐标系不一致（或 bbox/栅格映射不一致）
- 用 “WGS84→GCJ-02” 或 “GCJ-02→WGS84” 变换做对照，选能显著提高 on-road ratio 的版本作为合同口径

---

## 2. 分阶段实施计划（按 ROI + 可证伪）

> 时间估计是工作量级，不是承诺；每阶段都有 Go/No-Go。

### Stage 1（1 周）：用 OSM 道路 mask 替换 count-based 可行域（先把 CUT 指标变干净）

**目标**
- 去掉 weak-map `count>=1` 的孔洞导致的假切墙。

**交付物（数据）**
- `data/raw/network/shenzhen_osm.pbf`（或等价 OSM 输入）
- `data/processed_*/osm_road_mask.npy`（与 400×800 grid 对齐，bool）
- （可选）`data/processed_*/nav_field_osm.npz`：包含 `road_mask` 与必要 metadata（避免覆盖 count）

**验证（必须）**
- 在同一批 windows 上跑 oracle 线段碰撞审计：
  - 期望：oracle 在 OSM mask 上 `CUT` 显著下降（目标：<1% 或接近 0）
- 说明：这一步只是在“修评估与可行域定义”，不宣称模型能力提升。

**失败回滚（Stage 1 No-Go）**
- 若 OSM mask 下 `CUT` 仍高：
  1) 检查道路中心线是否做了足够 buffer（road width / dilation）
  2) 检查坐标系一致性（WGS84 vs GCJ-02）与 bbox 映射
  3) 检查 road-type 集合（service/unclassified 是否误排除）
  4) 兜底：先用 “hole-filled count mask（形态学 closing + dilation）” 作为过渡可行域，继续 Stage 2/3 的建模验证

### Stage 2（1–2 周）：引入道路拓扑信号，压 corridor error

**目标**
- 把 `corridor_error` 从 ~20% 压到 <10%（优先），让 end 能选对走廊。

**最小可行输入（KISS）**
- 以 OSM 道路为图：
  1) 构建连通图（路段/路口）
  2) 给定 dest，把“到 dest 的拓扑最短路距离”栅格化为 patch 通道（destination-conditioned）

**拓扑距离计算算法（reviewer 要求写清楚）**
- 距离度量：**路段长度加权**（grid/米均可，但需在合同里写清单位）
- 算法：Dijkstra（小图/局部图）  
- 栅格化方式（KISS 版本）：
  1) 像素中心点 → snap 到最近道路段（或最近路网节点）
  2) 用 snap 后的道路段/节点作为源点，在路网图上求到 dest 的最短距离
  3) 将该距离写回该像素，得到 `topo_dist_to_dest`（可做 clip + min-max 归一化到 [0,1]）
- 实用优化：只在 patch 覆盖的道路子图上做 Dijkstra（避免全图每样本过慢）

**交付物（特征）**
- `topo_dist_to_dest`（建议归一化后作为 nav_patch 额外通道，dest-centered 或 current-centered）

**验证（必须）**
- `end_imprecision_audit`：corridor/both 的占比显著下降
- `mask_alignment`：end 的 `JSD_pref` 明显下降（远离随机）

**失败回滚（Stage 2 No-Go）**
- corridor_error 不降：
  1) 检查 snap 是否合理（像素到道路距离阈值、最近邻错误）
  2) 检查 dest 是否在同一连通分量（连通性断裂会导致距离场退化）
  3) 检查距离归一化是否把信号“抹平”（过度 clip）
  4) 兜底：改为 dest-centered patch（让目的地附近走廊信息更清晰），或在 patch 内做局部 BFS/可达性而不是全局 Dijkstra

### Stage 3（2 周）：引入城市语义（POI/功能区/建成环境），提升绕路幅度（dev/len）

**目标**
- 把 under-detour 拉回：`max_dev_ratio/len_ratio` 的方向性偏差显著缩小。

**关于“POI vs 功能分区”的统一口径（避免重复数据源）**
- POI 是“点”，功能分区是“面”。建议从 POI 聚合出功能分区（主导类型 + 混合度/熵）。
- 功能分区也可以用卫星图像学到的连续特征来表示（建成环境/形态），但这是更高成本的增强层。

**卫星图像的位置（reviewer 要求明确）**
- 默认 **不做**（不进入主线），作为 Stage 3 的可选增强（ablation）：
  - 若 POI/功能区对 `dev/len` 无增益，再考虑引入卫星特征（建成环境、道路密度、绿地/水体）。
  - 数据源未拍板前不写死（Sentinel-2/商业影像/本地缓存等），避免工程被数据获取卡死。

**数据源一致性（必须讨论并写入合同）**
- 你手上有腾讯/高德 POI：两者分类体系、覆盖与更新周期不一致。
- 建议起步策略（KISS，先跑通再争论）：
  1) 先用 **单一数据源** 做主线（时间更接近轨迹/坐标更易对齐的那个）
  2) 只用 **一级粗分类**（餐饮/购物/办公/住宅/交通/医疗/教育/休闲…），降低不一致风险
  3) 生成 `poi_density_{cat}` 多通道栅格 + `landuse_dom`（主导类型）+ `landuse_entropy`（混合度）
  4) 后续再做双源融合与去重（作为 ablation，不先做主线）

**验证（必须）**
- `detour_scalar_direction_audit`：`Δp50 dev/len` 明显向 0 靠近（减少“更直更短”）
- `mask_alignment`：end 的 `JSD_pref` 下降（选点更接近 GT_proj）

**失败回滚（Stage 3 No-Go）**
- under-detour 不改善：
  1) 优先检查 POI 的时间一致性（2011 vs 2024 的偏差可能导致“伪相关”）
  2) 改用“轨迹自举语义”：用 2011 轨迹的 OD 密度/停留密度推断功能区（时间一致、成本低）
  3) 再考虑卫星图像/建成环境特征作为补充

### Stage 4（2 周）：Diffusion 多模态路线（在“单条路线够好”之后）

**定位**
- Diffusion 不再负责“把点放到路上/选对走廊”，这些由 OSM+拓扑+语义+AR 保证。
- Diffusion 负责“多种路线风格/意图”。

**推荐结构（更清晰、更可归因）**
- `Diffusion(路线意图 z)` → `AR(z, topo, semantic)` → `wp1/wp2/end` → `DetRes`

**验证**
- 多模态覆盖（多样性）与真实性（G2）同时不掉（避免“可行但无聊”的退化）。

---

## 3. 评估口径（避免再被“指标污染”坑）

### G1（可行性）
- drivable 定义以 **OSM road mask** 为准（而不是 count>=1）。
- `CUT` 必须在同一 mask 上计算，否则没有意义。

### G2（真实性）
主报告建议同时给三组证据：
1) **方向性**：`detour_scalar_direction_audit`（更直还是更绕）
2) **end 类型**：`end_imprecision_audit`（corridor vs dist）
3) **mask 内分布**：`macro_mask_alignment`（避免“mask 内乱选”）

---

## 4. 当前阶段裁决（今天该做什么）

> 先不写模型大改动，先让“可行域”变成真实道路。

- **立即推进 Stage 1：OSM mask 替换**  
  这是最小代价、最高收益：它会把 CUT 从“被孔洞抬高的 proxy 指标”变成“真实切墙指标”。

---

## 5. AR 结构说明（reviewer 要求：写清楚“AR 怎么做的”）

当前 AR 在代码里的实现是：
- **共享同一个 backbone**（2D CNN + FiLM 条件化），每一步输出一个 `(K×K)` heatmap logits
- 用 `prev_maps`（wp1_map/wp2_map 的 one-hot）作为额外输入通道，做三次顺序 forward：
  1) `p(wp1 | obs, OD, topo/semantic)`（prev_maps=0）
  2) `p(wp2 | obs, OD, topo/semantic, wp1)`（prev_maps=wp1 one-hot）
  3) `p(end | obs, OD, topo/semantic, wp1, wp2)`（prev_maps=[wp1, wp2] one-hot）

对应实现：`src/models/macro/macro_hardsupport_ar.py`

> 结论：Stage 2/3 引入的新通道（拓扑/语义）不需要改 AR 结构，只需要扩展 `nav_patch` 通道并在每一步共享使用。

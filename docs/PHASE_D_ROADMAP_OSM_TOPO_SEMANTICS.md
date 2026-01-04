# Phase D 路线图（OSM 道路先验（软） + 拓扑 + 城市语义 + AR + Diffusion 多模态）

> **这是一次方向转折点**：我们从“用 hard support 把输出裁进可行域（但切割真实分布）”转向“把地图/语义作为**软先验特征**进入模型，让模型自己学会‘更像真的’”。  
> **核心原则**：不把任何外部数据当作绝对真值；所有外部信息都必须以“开关消融 + 审计对照”方式进入系统，避免数据源黑箱。

---

## 0. 基于最新审计的精确诊断（为什么必须转向）

> 这里不讲符号，只讲“指标在说什么”。

> 注：本节引用的若干数值型证据来自 **legacy 深圳 dt30 的 Phase C pilot**（见 `docs/archive/legacy_shenzhen/PHASE_C_RESULTS.md` 及其审计产物），用于证明 failure mode（under-detour / 指标被 proxy 孔洞污染 / corridor error）在我们系统中确实发生过。  
> Detroit（WorldTrace×Detroit，1Hz）将复用同一套审计脚本与口径，但数值不会直接沿用，需在 Detroit 数据上重新生成。

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

- **OSM 道路先验（soft road prior）**：回答“哪里更像路”（不是禁止/裁剪），作为输入特征与软约束信号；同时用于生成拓扑图与审计口径之一。  
  - 明确：**不在训练环节用 OSM mask 做 hard cut / masked softmax**（避免切割真实分布、避免把模型上限绑定到地图质量）。
- **道路拓扑（topology feature）**：回答“哪条走廊更可能可达/更接近目的地”（corridor selection），以距离场/可达性等形式作为输入特征。
- **城市语义（POI/功能区/建成环境 feature）**：回答“为什么要绕、绕到哪里更像真的”（绕路动机/语义一致性），以多通道栅格/embedding 的形式作为输入特征。
- **AR 解码（结构机制）**：回答“怎么选才能连贯”（wp2 依赖 wp1，end 依赖 wp1+wp2；连通性写回模型）。
- **Diffusion（多模态）**：回答“有哪些不同路线风格”（前提是单条路线质量已足够）。
- **Micro DetRes（执行层）**：回答“怎么走得像真的”（局部动力学与形状质感）；不负责宏观绕路拓扑。

一句话：**AR 是“怎么选”的机制；拓扑/语义是“选什么”的信息；Diffusion 是“多样性”的来源。**

---

## 1.2 需要写进“数据契约”的变量（不靠拍脑袋：全部进入消融与审计）

> 你不希望用“优先级叙事”掩盖黑箱，因此这里不写“先做什么”，只写“必须记录什么、怎么审计”。

### A) 轨迹数据时间范围（决定 OSM/POI 的时间一致性风险）

- **轨迹时间范围（数据卡）**：`2021-08 ~ 2023-12`（WorldTrace）
- **试点城市**：Detroit（WorldTrace 子集，主坐标使用 `matched_latitude/longitude`，dt=1Hz）
- **影响**：
  - OSM/POI 若使用 2024 版本，存在“城市形态变化”的风险（尤其 POI 更严重）。
  - 道路网络相对更稳定，但 OSM 仍可能存在“缺路/偏移/中心线栅格化孔洞”；因此 OSM 也必须做审计对照。

### B) OSM 道路类型集合（必须作为可配置开关记录）

建议把 road-types 写成两套可选集合（A/B），并在报告里固定记录：
- **Set A（更保守）**：`motorway, trunk, primary, secondary, tertiary, residential`
- **Set B（更完整）**：Set A + `service, unclassified`

**明确排除（非机动车/步行）**：
- `footway, cycleway, path, steps`（以及类似的 pedestrian-only 类别）

> 注：OSM 是中心线，栅格化到 Detroit core `1024×1024`（约 25m/像素；见 `docs/DATA_CONTRACT.md`）时仍必须做“道路宽度 buffer/距离场”，否则会出现孔洞与对齐误差；但我们只把它当 **soft prior/审计 proxy**，不做训练期 hard cut。

### C) POI 主数据源与时间戳（必须记录与对照）

你手头有腾讯与高德 POI，但两者分类体系/覆盖/更新时间不同。为避免“数据源黑箱”，这里建议做两类对照：
- **单源对照**：Amap-only vs Tencent-only（都统一到一级粗分类后再栅格化）
- **融合对照（可选）**：Amap+Tencent（去重 + 统一映射）

### D) 坐标系/投影对齐（必须先审计，不许写死）

WorldTrace / OSM / SafeGraph 都以 **WGS84** 提供经纬度，因此主要风险不是 GCJ 偏移，而是：
- bbox / grid 映射口径错误（min/max 写反、`y` 轴方向、越界处理不一致）
- raw vs matched 坐标混用（`matched_distance` 分布异常，或 `is_matched` 门控失效）
- 米制距离换算口径不一致（`road_prob`/距离变换必须用 `res_x_m/res_y_m` 写死；见 `docs/DATA_CONTRACT.md`）

**不拍脑袋结论**：每次换数据源/换栅格，都必须跑一次“对齐审计”：
- matched vs raw 的 `matched_distance` 分布与 `is_matched` 比例（是否合理）
- 同一批点在 `osm_proxy` 下的 on-road proxy（buffer/dilation 敏感性必须输出）

---

## 2. 消融矩阵（可归因：每次只动一个开关）

> 目标：你担心的“数据源黑箱”本质是**因果归因不清**。  
> 解决方式不是“再加一个模块”，而是把每个新增信息源都做成**开关**，并用最小对照集回答：到底是谁带来的改善/退化。

### 2.1 需要显式建模的 5 个信息源（开关）

1) **道路先验（soft road prior）**：`count` 派生的密度/距离 proxy vs OSM 派生的 `road_prob/dist_to_road`  
   - 两者都只作为 **输入特征/软正则/审计 proxy**，不做训练期 hard cut（不 masked softmax）。  
   - `road_prob` 的默认定义与超参（`sigma=50m`）写死在 `docs/DATA_CONTRACT.md`。  
2) **道路拓扑（topo）**：`topo_dist_to_dest`（destination-conditioned 距离场）  
3) **城市语义（semantic）**：POI/功能区（从 POI 聚合得到的“面语义”）  
4) **多模态（multimodal）**：Diffusion 只负责“路线风格/意图”，不再承担“把点放到路上/选对走廊”的责任
5) **数据增强（ATR/STM）**：来自 UniTraj 的数据级增强（与地图/语义无关），必须独立消融（避免归因混淆）

> 这 4 个开关要做到：同一套训练脚本/评估脚本，只改配置就能切换输入；否则很容易在工程细节里引入不可见偏差。

### 2.2 最小可归因对照集（推荐直接按表跑，避免“讲故事”）

| Exp | road prior（soft） | topo_dist | POI/功能区 | 目的 |
|---|---|---|---|---|
| E0 | count(strict) | off | off | 复现当前基线（用于对照；只作为 proxy/特征，不裁剪） |
| E1 | count(hole-filled) | off | off | 把“孔洞效应”与“真实道路效应”拆开（只修 proxy 孔洞，不引入外部道路） |
| E2 | OSM road_prob | off | off | 验证：换成 OSM 道路先验后，CUT 是否更“干净”（oracle + dilation/buffer 敏感性） |
| E3 | OSM road_prob | on | off | 验证：topo_dist 是否降低 corridor error（不引入语义） |
| E4 | OSM road_prob | on | on | 验证：语义是否能拉回 under-detour（在 topo 已在场的前提下） |

> ATR/STM 是正交开关：建议对每个 E0–E4 都跑一次 `{ATR off/on} × {STM off/on}`，确保收益不被误归因到 road prior/topo/semantic。

**每个 Exp 都必须跑同一套审计输出**（否则不可比）：
- `oracle_cut_cause_audit`：看 CUT 是否被 mask 孔洞污染（oracle + dilation 敏感性）
- `macro_waypoint_gate`：G1（COLL/CUT/WP_ANY + seg0/1/2）
- `mask_alignment`：mask 内分布对齐（尤其 end 的 JSD_pref 是否远离随机）
- `end_imprecision_audit`：corridor/dist/both 的占比（定位 end 不精的具体类型）
- `detour_scalar_direction_audit`：`Δp50 max_dev_ratio/len_ratio`（方向性：更直还是更绕）

> 解释：这套对照集的目的不是“谁更好看”，而是回答“哪个数据源解决了哪个失败模式”。

### 2.3 各信息源的“最小实现”与失败回滚（不是先后关系，是可证伪）

#### OSM 道路先验（soft prior，不做 hard cut）
**交付物（数据）**
- `data/raw/network/detroit_osm.pbf`（或等价 OSM 输入；文件名按城市命名，避免混淆）
- `data/processed_*/osm_road_mask.npy`（与 Detroit core `1024×1024` grid 对齐，bool；仅作为特征/审计 proxy）
- `data/processed_*/osm_dist_to_road_m.npy`（float；到最近道路的距离，单位米；用于生成 `road_prob`）
- `data/processed_*/osm_road_prob.npy`（float；默认 `exp(-dist_to_road_m/50m)`；作为 soft prior 特征）
- （可选）`data/processed_*/osm_features.npz`：打包上述字段与 metadata（避免覆盖 `nav_field.count`）

**训练期口径（端到端，软约束）**
- 只允许 soft prior，不做 hard cut / masked softmax
- `L_offroad = λ_road * Σ(1 - road_prob(wp_i))`，`λ_road=0.1`（写死在数据契约/训练日志里）

**审计（必须）**
- **mask 质量审计（不等于真值）**：在同一批 windows 上跑 `oracle_cut_cause_audit` 的 dilation 敏感性：
  - 若轻微 dilation 能消掉大量 CUT（例如 >80%），说明 mask 有孔洞/中心线过细，需要调整 buffer/dilation
- **映射审计**：on-road proxy 的 buffer/dilation 敏感性 + OOB 比例（用于发现 bbox/grid 映射错误）
- 说明：以上审计只是在“评估/先验通道的质量控制”，不宣称模型能力提升。

**失败回滚（可证伪）**
- 若 OSM proxy 下 `CUT` 仍高：
  1) 检查道路中心线是否做了足够 buffer（road width / dilation）
  2) 检查 bbox/grid 映射是否一致（min/max、y 轴方向、越界处理）
  3) 检查 road-type 集合（service/unclassified 是否误排除）
  4) 兜底：对 OSM mask 做 buffer/dilation/closing 版本化（`osm_mask_v1/v2`），并把原始 OSM 与改良版同时作为审计口径输出，避免“偷偷换口径”。

#### 道路拓扑（topo_dist_to_dest）
**定义（KISS，reviewer 口径）**
- 距离度量：路段长度加权
- 算法：Dijkstra（局部子图版本：只在 patch 覆盖子图上算，避免全图过慢）
- 栅格化方式：
  1) 像素中心点 → snap 到最近道路段（或最近路网节点）
  2) 用 snap 后的道路段/节点作为源点，在路网图上求到 dest 的最短距离
  3) 将该距离写回该像素，得到 `topo_dist_to_dest`（可做 clip + min-max 归一化到 [0,1]）

**验证（必须）**
- `end_imprecision_audit`：corridor/both 的占比显著下降
- `mask_alignment`：end 的 `JSD_pref` 明显下降（远离随机）

**失败回滚（可证伪）**
- corridor_error 不降：
  1) 检查 snap 是否合理（像素到道路距离阈值、最近邻错误）
  2) 检查 dest 是否在同一连通分量（连通性断裂会导致距离场退化）
  3) 检查距离归一化是否把信号“抹平”（过度 clip）
  4) 兜底：改为 dest-centered patch（让目的地附近走廊信息更清晰），或在 patch 内做局部 BFS/可达性而不是全局 Dijkstra

#### 城市语义（POI/功能区/建成环境）

**目标**
- 把 under-detour 拉回：`max_dev_ratio/len_ratio` 的方向性偏差显著缩小。

**关于“POI vs 功能分区”的统一口径（避免重复数据源）**
- POI 是“点”，功能分区是“面”。建议从 POI 聚合出功能分区（主导类型 + 混合度/熵）。
- 功能分区也可以用卫星图像学到的连续特征来表示（建成环境/形态），但这是更高成本的增强层。

**卫星图像的位置（reviewer 要求明确）**
- 默认关闭（作为“语义模块”的可选增强开关，ablation）：
  - 若 POI/功能区对 `dev/len` 无增益，再考虑引入卫星特征（建成环境、道路密度、绿地/水体）。
  - 数据源未拍板前不写死（Sentinel-2/商业影像/本地缓存等），避免工程被数据获取卡死。

**数据源一致性（必须讨论并写入合同）**
- 你手上有腾讯/高德 POI：两者分类体系、覆盖与更新周期不一致。
- 默认配置（KISS；同时把“数据源选择”做成显式开关，避免黑箱）：
  1) **单源开关**：Amap-only vs Tencent-only（两者都统一到一级粗分类后再栅格化；结果必须成对报告）
  2) **融合开关（可选）**：Amap+Tencent（去重 + 统一映射；作为 ablation，不替代单源对照）
  3) 栅格产物：`poi_density_{cat}` 多通道 + `landuse_dom`（主导类型）+ `landuse_entropy`（混合度）

**验证（必须）**
- `detour_scalar_direction_audit`：`Δp50 dev/len` 明显向 0 靠近（减少“更直更短”）
- `mask_alignment`：end 的 `JSD_pref` 下降（选点更接近 GT_proj）

**失败回滚（可证伪）**
- under-detour 不改善：
  1) 检查 POI 的时间一致性（WorldTrace 2021–2023 vs POI 的年份，偏差可能导致“伪相关”）
  2) 改用“轨迹自举语义”：用同一时间段的轨迹密度/停留密度推断功能区（时间一致、成本低）
  3) 再考虑卫星图像/建成环境特征作为补充

#### Diffusion 多模态路线（在单条路线质量足够后再引入）

**定位**
- Diffusion 不再背“把点放到路上/选对走廊”的责任：这应由 **AR 的结构约束 + topo/语义特征 + OSM 软先验**共同学习出来（而不是靠 hard cut 保证）。
- Diffusion 负责“多种路线风格/意图”。

**推荐结构（更清晰、更可归因）**
- `Diffusion(路线意图 z)` → `AR(z, topo, semantic)` → `wp1/wp2/end` → `DetRes`

**验证**
- 多模态覆盖（多样性）与真实性（G2）同时不掉（避免“可行但无聊”的退化）。

**Diffusion 触发门槛（你已拍板；必须同时满足）**
- `CUT < 5%`（OSM 口径；并附 buffer/dilation 敏感性，避免 proxy 孔洞污染）
- `corridor_error < 10%`（end 选错平行走廊的比例）
- `Δp50(dev/len) < 0.1`（方向性审计：预测不再系统性偏直/偏短）

**跨城市验证预留（E5）**
- `E5: Chicago/NYC zero-shot`（预留；必须复用同一套审计口径与数据契约字段）

---

## 3. 评估口径（避免再被“指标污染”坑）

### G1（可行性）
- **不允许只报一个口径**。为了避免“mask 黑箱”，G1 必须同时输出两套代理口径（sensitivity audit）：
  - `count_proxy`：`count>=thr` 及其 hole-filling（closing/dilation）的敏感性
  - `osm_proxy`：OSM mask 及其 buffer/dilation 的敏感性
- `CUT` 必须与对应的 mask/proxy 配套计算，并在 JSON 里写明使用的 proxy 版本号（否则不可比）。

### G2（真实性）
主报告建议同时给三组证据：
1) **方向性**：`detour_scalar_direction_audit`（更直还是更绕）
2) **end 类型**：`end_imprecision_audit`（corridor vs dist）
3) **mask 内分布**：`macro_mask_alignment`（避免“mask 内乱选”）

---

## 4. 执行口径（每次实验都必须输出）

> 为了避免“指标被 proxy 污染导致归因失效”，所有 topo/语义对照都必须 **同时输出两套 proxy 口径**（count vs OSM），并附上 dilation/buffer 敏感性。  
> 这不是“先后顺序”，而是保证每个结论 **可归因、可复现、可对照**。

---

## 5. AR 结构说明（reviewer 要求：写清楚“AR 怎么做的”）

仓库中已验证的 AR 基线（Phase C）：
- **共享同一个 backbone**（2D CNN + FiLM 条件化），每一步输出一个 `(K×K)` heatmap logits
- 用 `prev_maps`（wp1_map/wp2_map 的 one-hot）作为额外输入通道，做三次顺序 forward：
  1) `p(wp1 | obs, OD, topo/semantic)`（prev_maps=0）
  2) `p(wp2 | obs, OD, topo/semantic, wp1)`（prev_maps=wp1 one-hot）
  3) `p(end | obs, OD, topo/semantic, wp1, wp2)`（prev_maps=[wp1, wp2] one-hot）
- Phase C 的实现文件：`src/models/macro/macro_hardsupport_ar.py`（注意：该基线包含 hard support/约束逻辑，用于“上界/诊断/止损”，不作为 Phase D 主线能力宣称）

Phase D 主线（端到端、软先验）对 AR 的要求：
- **保留相同的 AR 结构**（三步条件化 + `prev_maps`），因为它已被验证能显著改善连贯性/连通性。
- **移除训练期 hard cut / masked softmax**：输出空间不再被外部 mask 裁剪；OSM 只通过 `road_prob` 特征 + `L_offroad` 软正则影响采样分布（详见 `docs/DATA_CONTRACT.md`）。

> 结论：Phase D 的新增通道（`road_prob/topo/poi/landuse`）不需要改 AR 的“顺序条件化骨架”，只需要扩展输入通道，并确保每一步共享同一套特征口径与日志审计。

---

## 6. UniTraj 借鉴（与地图无关，作为可开关增强）

> 这两项不依赖 OSM/POI，不会引入“地图质量绑定”的问题；适合做为稳定性/泛化的 data-level 增强，并进入消融矩阵。

### 6.1 Adaptive Resampling（对数重采样）
- 目的：不同长度/不规则采样的轨迹，信息量不是线性增长；对数重采样能压缩冗余长轨迹、保留短轨迹细节。
- 形式：在数据加载/窗口抽取时，对每条轨迹按长度自适应采样率重采样（不改变坐标系）。

### 6.2 Self-supervised Masking（多策略遮盖）
- 目的：让 encoder 学到“缺失/断裂时仍能恢复结构”，提升对噪声与掉点的鲁棒性。
- 形式：random/block/keypoint/last-N 四类遮盖，训练重建被遮盖点（或与预测任务共享 backbone）。

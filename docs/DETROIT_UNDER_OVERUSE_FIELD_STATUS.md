# Detroit under/over-use 场阶段性报告（发给 PI）

> 版本：2026-01-xx（以工作站A最新输出为准）  
> 说明：深圳数据已封存，本阶段不涉及深圳分析，仅讨论 WorldTrace（Detroit/Columbus）链路。

## 一句话主线（统一口径）
**全局“绕不绕”并不能区分城市断裂；断裂的 signature 只能在“路线选择的空间组织”里出现，因此我们要构造一个可校准的行为参照系，并把“实际 vs 参照”的差异空间化为 under/over-use 场。**

---

## 0. 现在推进到叙事主线的哪一步？
我们把叙事拆成 5 个必须按顺序“过闸”的阶段（任何一步不过，后面的图都不能升级叙事）：

1. **数据对齐完成**（WorldTrace 轨迹段、OSM、POI、卫星影像、Census tract）  
2. **全局标量 null result**：Detroit vs Columbus 的 detour 标量差异不显著（用于动机：必须做空间具体性）  
3. **物理参照系证伪**：最短路/可达性 baseline 会产生大量伪 under-use（用于动机：需要行为参照而非物理参照）  
4. **行为参照系自校准（Self-check）**：Columbus→Columbus 参照应“对齐走廊 mode”，否则 Detroit 图没有可解释性  
5. **Detroit 结果解释与外部验证**：在自校准通过后，才讨论 Detroit 的 under/over-use 场与 vacancy/income 等指标的空间对应

**当前卡在第 4 步：Self-check 未通过，因此 Detroit 的图只能作为“未校准的 under/over-use 可视化”，不能称为 avoidance。**

---

## 1) 阶段性事实（目前我们已经能“确定”的）

### 1.1 数据对齐已经足够支撑后续分析（Step 1）
工作站A的 Detroit/Columbus 都已完成：
- 轨迹段抽取（WorldTrace）  
- OSM road probability（soft prior）  
- SafeGraph POI 栅格（一级粗分类）  
- Wayback 影像瓦片（z=16，多 release）  
- Census ACS tract 协变量（vacancy / income / population），并完成 Detroit bbox 过滤与清洗

这一步解决的是“数据是否足够/一致”的问题：**目前没有证据表明问题来自数据缺失或坐标错配。**

### 1.2 全局 detour 标量不能区分 Detroit（Step 2：null result）
在当前 bbox 与窗口口径下：
- Columbus：`len_ratio_p50 ≈ 1.296`，`max_dev_ratio_p50 ≈ 0.197`  
- Detroit：`len_ratio_p50 ≈ 1.297`，`max_dev_ratio_p50 ≈ 0.166`  

含义（叙事价值）：  
**“断裂城市更绕”这个直觉假设被数据否定。**  
如果断裂存在 signature，它不在全城平均 detour，而在“哪些走廊被用/被替代”的空间结构里（这正是我们要做 under/over-use 场的动机）。

### 1.3 物理参照系（最短路/可达性）不是行为 baseline（Step 3）
在 Columbus 的物理参照实验里，expected 的空间覆盖（support）远大于人真实会走的走廊集中（mode）。  
直观结果：用物理参照做 `log(obs/exp)` 会在任何城市都产生大量负值（under-use），属于伪信号。

叙事意义：  
**我们不是要比较“可达性 vs 实际选择”，而是要比较“正常行为选择 vs 断裂城市选择”。因此必须构造行为参照系。**

---

## 2) “Expected footprint”到底是什么？为什么它是参照系？
这里必须用非符号语言讲清楚，否则审稿人会质疑“你到底在比较什么”。

### 2.1 Observed footprint（实际足迹）
把 Detroit（或 Columbus）里每条轨迹段在网格上“踩过”的格子计数并归一化，得到一个空间概率分布：  
**某个格子值越大，表示真实车流越集中经过这里。**

### 2.2 Expected footprint（参照足迹）
Expected 的目标不是“所有可能会走的路”（support），而是“正常情况下最可能走的少数走廊”（mode / major modes）。  
在我们的设定里：
- **行为参照系**：从功能正常城市（Columbus）的真实路线选择中学出“同类 OD + 同类时间段”下的主走廊选择，并把它投影成 expected footprint。  
- **OSM road_prob 只做可行性 prior**：它只回答“这里有路/能走”，不回答“大家偏好走哪”（偏好必须来自 Columbus 的真实频率）。

### 2.3 under/over-use 场（暂不叫 avoidance）
对每个格子计算 `log(observed / expected)`（或相对差），得到空间差异场：
- **负值**：相对参照“少走”（under-use）  
- **正值**：相对参照“多走”（over-use）  

在 self-check 通过前，我们只使用 under/over-use 这个描述性术语，不做“回避”的因果解释。

---

## 3) 当前最大的科学风险：参照系尚未校准（Self-check 未过）
我们已经按 PI 建议把“模板端”收敛到了 mode/major-modes（有明确的 mode_frac 指标），但 **landing（落地）仍把 mode 扩散成 support**。

### 3.1 现象：Expected 覆盖面仍显著大于 Observed
以 Columbus self-check 最新一轮（peak3 + major modes + corridor penalty）为例：
- `obs_cells_ratio ≈ 0.0668`（真实轨迹占用格子约 6.7%）  
- `support_cells_ratio ≈ 0.2763`（expected 占用格子约 27.6%）  
- `mass_exp_in_obs_cells ≈ 0.314`（expected 质量只有约 31% 落在真实走过的格子上）

含义：expected 仍然“铺得太开”。这会导致：
- `log(obs/exp)` 在 support 内大量为负（看起来像大面积 under-use）  
- 这种负值很可能是参照系偏差，而非城市断裂信号

### 3.2 当前 Detroit 图的定位（必须诚实）
目前已经生成了 Detroit/Columbus 的可视化（例如：  
`essay/figures/worldtrace_detroit/avoidance_ref/avoidance_log_ratio.png`、  
`essay/figures/worldtrace_columbus/avoidance_ref/avoidance_log_ratio.png`），但：

**在 self-check 没通过前，这些图只能用来说明“物理参照不够、行为参照需要校准”，不能作为 Detroit avoidance 的主结果图。**

---

## 4) 下一步要做什么（保证叙事推进，而不是调参循环）
我们现在要做的不是“再画更多 Detroit 图”，而是让参照系先变得可校准。

### 4.1 必须完成的门槛：Self-check 通过
在 Columbus→Columbus 下，expected 必须表现为“少数走廊集中”，而不是“在路网里散开”。

建议的判据（用于内部 gate；最终阈值可由 PI 决定）：
- `support_cells_ratio` 明显收缩（接近 observed 的 1–2 倍，而不是 4 倍）  
- `mass_exp_in_obs_cells` 明显上升（至少不应长期卡在 ~0.3）  
- 图像层面：主走廊的亮带应与 observed 对齐（定性必要条件）

### 4.2 当前技术瓶颈的本质（给 PI 的一句话）
**模板已经是 mode 了，但 A* landing 在 corridor buffer 内“自由扩散”，把 mode 变成 support。**

因此下一步的技术动作应围绕“让 landing 对 corridor 有收缩力”展开（而不是继续改 OSM 权重把偏好硬塞进先验）。

---

## 5) 需要 PI 讨论/拍板的开放问题（如果要迅速收敛）
1. **Self-check 的通过标准到底用什么？**  
   我们是否以“走廊对齐的定性检查 + 若干统计指标”联合判定？阈值是否要 city-size 归一化？
2. **行为参照的定义是否允许 2–3 个 major modes mixture？**  
   在通勤场景双峰很常见，强行单模态可能会误伤正常多样性。
3. **如果 landing 仍无法收敛，是否允许 pivot 到“直接用真实 footprint 做 expected”？**  
   这将显著缩短叙事链条（matching+比较），但会弱化“生成模型作为测量工具”的成分；是否符合投稿定位需要 PI 决策。

---

## 6) 本阶段可给子刊叙事的“确定性结论”（可以写进正文的）
1. **全局 detour 标量不足以区分 Detroit vs Columbus（null result）**：这迫使我们转向“空间具体性”。  
2. **物理参照（最短路/可达性）会产生系统性伪 under-use**：它衡量的是“可达但未走”，不是“正常应走但被回避”。  
3. **行为参照必须先通过 self-check 才能解释 Detroit**：否则任何 under/over-use 场都可能是参照系偏差。

---

## 附：目前可用于 PI 快速查看的图（本地路径）
- Detroit story：`essay/figures/worldtrace_detroit/story/`  
- Detroit under/over-use（未校准，仅展示）：`essay/figures/worldtrace_detroit/avoidance_ref/`  
- Columbus self-check（用于校准诊断）：`essay/figures/worldtrace_columbus/avoidance_ref/`


# Data Contract（数据契约｜WorldTrace × Detroit）

> 目的：把“主数据/语义数据/地图数据/版本/坐标系/落盘形态”写成单一真相源，避免后续出现“数据不一致导致指标变化却无法归因”。  
> 原则：能写死的写死；不确定的写成 `TBD`，并给出**审计方法与通过标准**；不把外部地图当真值。

---

## 0) 当前采用的数据配置（你已拍板）

- 主数据：`OpenTrace/WorldTrace`（UniTraj 数据底座）
- 试点城市：Detroit（美国）
- 训练坐标：使用 `matched_latitude/longitude`
- 时间分辨率：1Hz（WorldTrace 标准化后）
- 空间栅格：Detroit core bbox + `1024×1024` grid（约 25m 分辨率；用于 OSM/POI/landuse 栅格化与 patch 提取）
- 地图数据：OSM 作为辅助特征（软先验/软约束），不用于训练期 hard cut
- 语义数据：SafeGraph POI + 遥感影像 + 土地利用（land use）
- 大规模处理：工作站 A 下载并转换（优先解决“文件数/IO”问题）

---

## 1) WorldTrace（主数据底座）

### 1.1 数据源与版本

- HuggingFace：`OpenTrace/WorldTrace`
- 时间范围（数据卡）：2021-08 ~ 2023-12
- 标准化采样：1 second（after standardization）
- 坐标系（数据卡）：WGS84
- 许可：ODbL 1.0
- 版本锁定（拍板）：
  - HF revision（commit SHA）：`8fb49c023b8a2571d4cbec07681e069477743eb5`
  - commit date：`2025-09-25`
- 大文件指纹（拍板；用于验明正身，避免未来 main 漂移）：
  - `Trajectory.zip`：
    - commit：`c7a8445775771498fb413e64f7e64a709643d93f`
    - git-lfs oid sha256：`3233f26eb87f6ba88f622224cb93b64bbe48bcbfcf8b57a2035ea016f2fe7693`
    - size_bytes：`27166178373`
  - `Meta.zip`：
    - commit：`5a1dd147dfe37227e3484f1ebef5b7bdc38859f6`
    - git-lfs oid sha256：`220ead3b4eabe233163ffed9270c56db29972d5eeac0e01c64421a6ab8b3cf14`
    - size_bytes：`7800430228`

### 1.2 轨迹字段口径（训练/审计）

轨迹 csv（字段来自数据卡与样例）：
- `time`
- raw：`latitude, longitude`（WGS84）
- map-matched：`osm_way_id, matched_latitude, matched_longitude, matched_distance, matched_type`

**训练坐标（已决策）**
- 使用：`(lat, lon) := (matched_latitude, matched_longitude)`
- 保留：raw 坐标仅用于审计（例如检查匹配偏差/异常轨迹）

**匹配质量（待审计写死）**
- 点级别策略（拍板）：
  - `MATCHED_DISTANCE_MAX_M = 30`
  - 若 `matched_type` 可用且 `matched_distance <= 30`：
    - 使用 `matched_latitude/matched_longitude`，并标记 `is_matched=1`
  - 否则：
    - fallback 到原始 `latitude/longitude`，并标记 `is_matched=0`
- 轨迹/段级别质量闸门（拍板；作用对象为 Detroit bbox 内“最长连续段”，见第 2 节）：
  - `MAX_UNMATCHED_RATIO = 0.20`（该段内 `is_matched=0` 比例 > 20% 则丢弃该段）
- `matched_type` 的语义（True/False 的含义）：`TBD`（需要全量统计确认后再写死；当前口径暂按“可用/不可用”二值门控执行）

### 1.3 元信息（Meta）

Meta json（字段来自数据卡与样例）：
- `Filename, Uploaded, Points, Start coordinate, End coordinate, Distance, Time, geometry, ...`
- `Owner/Description/Tags/Visibility`：默认不入训练（隐私/偏置风险），仅用于审计或数据清洗 debug。

---

## 2) Detroit 子集（可复现筛选规则）

WorldTrace 不提供城市标签，因此 Detroit 必须用空间规则筛选。此处必须写死：

### 2.1 Detroit official boundary bbox（用于“城市粗筛/审计对照”，拍板）

- Detroit 城市边界与 bbox 来源（拍板）：
  - ArcGIS Feature Layer：`City_of_Detroit_Boundary`
  - url：`https://services2.arcgis.com/qvkbeam7Wirps6zC/arcgis/rest/services/City_of_Detroit_Boundary/FeatureServer/0`
  - extent_epsg3857（102100/3857）：
    - XMin：`-9271556.12220758`
    - YMin：`5199248.13369315`
    - XMax：`-9229537.24`
    - YMax：`5228684.0021`
  - bbox_epsg4326（WGS84；由上述 extent 3857→4326 换算得到）：
    - min_lon：`-83.287806`
    - min_lat：`42.254960`
    - max_lon：`-82.910344`
    - max_lat：`42.450375`

### 2.2 Detroit core training window（用于“栅格化/patch/训练”，拍板）

> 说明：这不是“城市边界真值”，而是为了把栅格定义写死（1024×1024 正方形，便于计算与跨源对齐）。

- bbox_epsg4326（WGS84；训练窗口）：
  - min_lon：`-83.25`
  - max_lon：`-82.95`
  - min_lat：`42.25`
  - max_lat：`42.50`
- grid：
  - `H = 1024`
  - `W = 1024`
- grid 索引约定（必须全项目一致）：
  - `y` 为行（0 在北，向南增大），`x` 为列（0 在西，向东增大）
  - `x = floor((lon - min_lon) / (max_lon - min_lon) * W)`
  - `y = floor((max_lat - lat) / (max_lat - min_lat) * H)`
  - 越界样本：`x∉[0,W-1]` 或 `y∉[0,H-1]` 直接视为 OOB
- cell 分辨率（米；用于距离变换/road_prob；拍板为“按 bbox 等距近似”）：
  - `res_x_m ≈ haversine((min_lat,min_lon),(min_lat,max_lon)) / W`
  - `res_y_m ≈ haversine((min_lat,min_lon),(max_lat,min_lon)) / H`
  - 备注：这是一阶近似（Detroit 范围内足够用）；更精确方案（EPSG:3857/UTM）作为 `TBD` 仅在需要时引入。

### 2.3 轨迹筛选口径（拍板：bbox 粗筛 + bbox 内连续段切片）

  - 坐标字段优先级（用于筛选与切片）：
    1) 若点满足“matched_type 可用”且 `matched_distance <= 30m`：使用 `matched_latitude/matched_longitude`（`is_matched=1`）
    2) 否则 fallback 到 `latitude/longitude`（`is_matched=0`）
  - bbox 内点过滤：按 `Detroit core training window bbox_epsg4326` 过滤出在 bbox 内的点
  - 连续段切分：若相邻点 `Δt > 5s` 则断开（WorldTrace 标准化后采样间隔为 1s）
  - 保留策略：保留 bbox 内**最长连续段**
  - 最小段长度：`min_segment_points = 120`（2 分钟，1Hz；低于 120 点丢弃）
  - 段级别质量闸门：该段内 `is_matched=0` 的比例 **> 20%** 则丢弃该段

通过标准（必须输出统计）：
- Detroit 子集轨迹条数、点数、时长/里程分布
- 与全量 WorldTrace 的对比（避免筛选造成系统性偏置）

---

## 3) 语义数据（Detroit）

### 3.1 SafeGraph POI

- POI 数据源：SafeGraph
- 产品：`Places (Base + Rich + Geometry)`
- 数据版本时间（vintage，拍板）：`SAFEGRAPH_PLACES_VINTAGE = 2024-01`（YYYY-MM）
- 坐标系（拍板）：`EPSG:4326 (WGS84)`
- 分类口径：统一到一级粗分类（餐饮/购物/办公/住宅/交通/医疗/教育/休闲…）
- POI 在 vintage 月是否有效（拍板）：
  - `poi_active(vintage)` 为真当且仅当：
    - (`opened_on` is null) OR (`opened_on` <= vintage)
    - AND
    - (`closed_on` is null) OR (`closed_on` > vintage)
  - 特例：若 `closed_on == '1900-01'`，按永久关闭处理，直接视为 inactive

输出建议（栅格化到同一坐标系/投影后）：
- `poi_density_<cat>`：float raster（密度）
- `landuse_dom`：主导功能类型（int / one-hot）
- `landuse_entropy`：混合度（float）

### 3.2 遥感影像

- 数据源：`TBD`（Sentinel-2 / 商业影像 / 本地缓存）
- 时间戳：`TBD`
- 分辨率：`TBD`
- 覆盖范围：必须覆盖 Detroit 范围 + buffer（避免边界效应）

### 3.3 土地利用（Land Use）

- 数据源：`TBD`
- 时间戳：`TBD`
- 类别体系：`TBD`（需要映射到统一口径）

---

## 4) OSM（辅助特征，不做 hard cut）

### 4.1 输入与版本

- OSM 数据来源：`TBD`（建议记录 extract 来源与时间戳）
- OSM 版本/快照时间：`TBD`

### 4.2 道路类型集合（可配置集合，进入消融）

- Set A：`motorway, trunk, primary, secondary, tertiary, residential`
- Set B：Set A + `service, unclassified`
- 排除：`footway, cycleway, path, steps`（以及 pedestrian-only）

### 4.3 作为特征的形式（候选）

- `road_mask`（中心线 buffer/dilation 后的 raster）
- `dist_to_road`（距离变换；更适合作为软先验）
- `road_prob`（软先验：连续概率场；默认定义见下）
- `topo_dist_to_dest`（拓扑/可达性距离场；Stage D 的 corridor 关键信号）

重要约束：
- 不把 `road_mask` 用于训练期 masked softmax/hard cut（避免切割真实分布与地图质量绑定）。
- `road_mask` 可以用于审计（例如 on-road ratio、cut/coll 的 proxy 口径对照）。

### 4.4 `road_prob` 定义（拍板：默认 A + 消融 B）

**A. 距离连续概率（默认）**
- `dist_to_road_m = distance_transform_edt(~road_mask, sampling=(res_y_m, res_x_m))`
- `ROAD_PROB_SIGMA_M = 50`
- `road_prob = exp(-dist_to_road_m / ROAD_PROB_SIGMA_M)`

**B. 二值膨胀概率（消融/兜底）**
- `road_prob = binary_dilation(road_mask, iterations=2).astype(float)`（0/1）

注意：
- A 的优势是梯度更平滑、对小幅 OSM 偏移更鲁棒；B 用于验证“连续概率是否必要”。

### 4.5 训练期 soft prior（端到端口径，拍板）

> 这里不做 hard cut，只加软正则，让模型自己学会把概率质量压回“更像路”的区域。

- `L_offroad = λ_road * Σ_i (1 - road_prob(wp_i))`
  - `wp_i`：Macro 的 waypoint（例如 wp1/wp2/end 或更长序列的采样点）
  - `λ_road` 初始：`0.1`（后续只允许在“记录+审计”前提下调整）
- 训练日志必须同时输出：
  - 主任务损失（例如 NLL/MSE）
  - `L_offroad` 与其占比（避免“约束项失效/被淹没”）

---

## 5) 落盘形态（必须解决“海量小文件 IO”）

WorldTrace 是海量小文件（每条轨迹一个 csv/json）。建议做两层产物：

1) **manifest**（必须）：从 Meta 解析出的全局索引表（parquet/arrow），支持 Detroit 筛选与抽样，不重复扫描轨迹文件。
2) **训练落盘**（二选一）：`parquet(partitioned)` 或 `HDF5/LMDB`（按你们训练栈选择；但必须固定记录）。

---

## 6) 评估与审计口径（避免“外部数据真值化”）

由于我们不做 hard support，G1/G2 需要明确“代理口径”：

- `matched_onroad_proxy`：利用 `matched_distance` 的阈值/分布做匹配质量审计（拍板：`matched_distance<=30m` 视为 matched，可用 `is_matched` 汇总 unmatched_ratio）
- `osm_proxy`：OSM 派生的 `road_mask`/`dist_to_road` 作为审计口径（版本化：buffer/dilation 参数必须记录）
- `trajectory_density_proxy`（可选）：Detroit 子集中统计得到的道路热力（避免完全依赖 OSM）

每次报告必须同时输出：
- 训练配置（哪些特征开关打开）
- 审计口径（用的是哪个 proxy 版本）

---

## 7) Legacy（旧数据保留用于复现，不再作为主线）

> 旧实验（Phase C）基于深圳出租车 GPS（Passenger Trip，dt=30s），用于复现历史结论。  
> 若仍需复现，请在旧机器/旧数据根目录下运行对应脚本；新主线已切换到 WorldTrace×Detroit（1Hz matched）。

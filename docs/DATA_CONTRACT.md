# Data Contract（数据契约｜Phase D）

> 目的：把“轨迹/OSM/POI/语义特征/坐标系/版本”写成单一真相源，避免后续出现“数据不一致导致指标变化却无法归因”的情况。  
> 原则：能填的填死；不确定的写成“待审计”，并给出审计方法与通过标准。

---

## 1) 轨迹数据（Trip trajectories）

- 数据集：深圳出租车 GPS（Passenger Trip）
- 时间范围（已知）：`2011-04-18 ~ 2011-04-26`
- 原始文件：`data/raw/gps/*.txt`（GBK CSV）
- 清洗合同（已拍板）：
  - `status == 1`（Passenger Trip）
  - `max_speed_kmh = 120`
  - `max_gap_s = 300`
  - 时区：北京时间（Asia/Shanghai, UTC+8）

### 坐标系（必须确认）

- 轨迹经纬度坐标系：`UNKNOWN (WGS84 vs GCJ-02)`  ← **待审计**

**审计方法（Stage 1 必做）**
- 构建 OSM road mask 后，统计轨迹点落在 road mask 上的比例（on-road ratio）：
  - 若 on-road ratio 很低（例如 <30%），优先怀疑坐标系不一致或 bbox/投影不一致
  - 对照：对轨迹坐标做 WGS84↔GCJ-02 变换后再统计

**通过标准**
- 选择能显著提高 on-road ratio 的坐标系/变换作为合同口径，并记录：
  - 采用的坐标系名称（WGS84/GCJ-02）
  - 变换方向与实现版本（代码文件/commit）

---

## 2) 栅格与空间范围（Grid / BBox）

- Grid：`H=400, W=800`
- BBox（当前仓库默认）：`lat[22.45,22.85], lon[113.75,114.65]`
- 单元分辨率（已测）：约 `111m × 115m`
- Patch：
  - `32×32` 覆盖约 `3.55km × 3.69km`
  - `64×64` 覆盖约 `7.10km × 7.38km`

---

## 3) OSM（道路网络）

### 输入文件与版本

- OSM 原始文件：`data/raw/network/shenzhen_osm.pbf`（建议命名；可替换为实际文件名）
- OSM 数据版本/快照时间：`UNKNOWN (YYYY-MM)` ← **待确认**
  - 说明：轨迹是 2011；理想情况使用接近 2011 的 snapshot，但工程上允许先用最新 OSM 做 Stage 1/2 baseline（需在论文/报告里写清局限）。

### “drivable road”的道路类型（必须写死）

- 默认纳入：`motorway, trunk, primary, secondary, tertiary, residential`
- 决策（reviewer 要求）：`service=YES, unclassified=YES`
- 明确排除：`footway, cycleway, path, steps`（以及 pedestrian-only 类别）

### 栅格化与道路宽度（必须记录）

- OSM road mask 栅格化方法：`UNKNOWN` ← **待实现后补齐**
  - 必须记录是否对道路中心线做 buffer（road width / dilation），否则会产生孔洞。

输出：
- `data/processed_*/osm_road_mask.npy`（bool，H×W）
- `data/processed_*/osm_road_graph.pkl`（可选；Stage 2 用）

---

## 4) POI（点兴趣）与功能分区（面语义）

### 主数据源

- POI 主源：`Amap (Gaode)`（决策）
- POI 备选补充：`Tencent`（可选）
- OSM POI：不作为主源（覆盖与分类不稳定），可作为补充类型（可选）

### 时间一致性（必须记录）

- POI 采集时间：`UNKNOWN (YYYY-MM)` ← **待填**

> 风险提示：轨迹是 2011；若 POI 是 2024，会存在“新建商场/新路网”导致语义错配的风险。  
> Stage 3 必须做可证伪 ablation：POI 是否真的改善 end/corridor 或 dev/len，而不是引入伪相关。

### 分类口径（KISS）

- 统一到一级粗分类（示例）：餐饮/购物/办公/住宅/交通/医疗/教育/休闲…
- 功能分区：从 POI 聚合生成（主导类型 + 混合度/熵 + 各类密度）

输出（建议）：
- `data/processed_*/poi_density_<cat>.npy`（float，H×W）
- `data/processed_*/landuse_dom.npy`（int，H×W）
- `data/processed_*/landuse_entropy.npy`（float，H×W）

---

## 5) 拓扑距离场（Stage 2）

- 距离度量：路段长度加权
- 算法：Dijkstra（局部子图优先）
- 像素到路网的 snap 策略：`UNKNOWN` ← 待实现后补齐
- 输出通道：`topo_dist_to_dest`（归一化到 [0,1]）

---

## 6) 卫星图像（可选增强，不进入主线）

- 是否启用：默认 `NO`（Stage 3 ablation）
- 数据源：`TBD`
- 时间戳：`TBD`

---

## 7) 结果与评估口径（必须对齐）

- G1/G2 的 drivable mask 口径：
  - Phase C：`count>=thr`（weak-map proxy）
  - Phase D：改为 `osm_road_mask`（主口径），并保留 count 作为偏好/强度通道（不是可行域定义）


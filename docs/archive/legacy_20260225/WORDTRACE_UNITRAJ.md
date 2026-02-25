# WorldTrace × UniTraj：数据底座与预训练策略（与本项目对接｜Detroit 试点）

> 目的：把 WorldTrace（数据底座）与 UniTraj（预训练范式）整理成**我们项目可执行的数据/训练契约**。  
> 原则：不靠“讲故事”，只写能落地、能审计、能消融的内容；不把外部地图当真值，不把 hard support 当能力。

---

## 0) 本次已拍板的决策（你的最新结论）

- 数据底座：`OpenTrace/WorldTrace` 作为**我们自己的预训练数据集**（也是 UniTraj 底座）。
- 试点城市：Detroit（美国）。
- 主坐标：使用 WorldTrace 的 **map-matched 坐标**：`matched_latitude/matched_longitude`。
- 时间分辨率：与 WorldTrace 一致（**1Hz**）。
- 语义数据：Detroit 的 SafeGraph POI + 遥感卫星图像 + 土地利用（land use）。
- 地图信息：OSM（以及 WorldTrace 自带的 `osm_way_id`）作为**辅助特征（软先验/软约束）**进入特征学习，不作为训练期硬裁剪/硬约束。
- 空间栅格：Detroit core bbox + `1024×1024` grid（约 25m/像素；用于 OSM/POI/landuse 栅格化与 patch 提取；口径见 `docs/DATA_CONTRACT.md`）。
- 大规模处理：在工作站 A 下载与转换（主要风险是“文件数/IO”而不是算力）。

---

## 1) WorldTrace 与 UniTraj 的关系（结论版）

- **WorldTrace 是数据集贡献**：全球覆盖、统一 1Hz、包含 OSM map-matching 信息的车载轨迹底座。
- **UniTraj 是方法贡献**：用 WorldTrace 做大规模自监督预训练，学通用轨迹表征（再适配任务）。

一句话：**WorldTrace 是“数据底座”，UniTraj 提供“预训练策略与工程经验”。我们复用的是：底座 + 预训练策略，不照搬“极简输入、完全不看地图语义”的哲学。**

---

## 2) WorldTrace 数据卡（HuggingFace README）关键信息

> 来源：`https://huggingface.co/datasets/OpenTrace/WorldTrace`

- 规模：2.45M trajectories；处理后约 880M GPS points（raw 约 8.8B points）。
- 覆盖：70 countries；时间跨度 2021-08 ~ 2023-12。
- 标准化：采样间隔统一到 1 秒。
- 坐标系：WGS84（raw `latitude/longitude`），并提供 map-matched 坐标与 OSM way id。
- 许可证：ODbL 1.0（后续要发布衍生数据/模型时必须提前审查合规路径）。
- HF 仓库文件：`Trajectory.zip`（csv），`Meta.zip`（json），以及样例 `trajectory_sample.csv`/`meta_sample.json`。

---

## 3) 轨迹 CSV / Meta JSON 字段（与本项目对接视角）

### 3.1 轨迹 CSV（每条轨迹一个 csv）

字段（README + 样例确认）：
- `time`
- `latitude`, `longitude`（WGS84）
- `altitude`（可选）
- `osm_way_id`
- `matched_latitude`, `matched_longitude`
- `matched_distance`
- `matched_type`（样例中为 True/False；语义需用全量统计确认）

我们试点（Detroit）的使用口径：
- 主坐标：`(lat, lon) := (matched_latitude, matched_longitude)`
- 保留 raw 坐标用于审计：`(raw_lat, raw_lon)`
- 保留 `matched_distance` 作为质量特征（可用于过滤/降权；阈值需审计后再写死）
- 保留 `osm_way_id` 作为“拓扑/道路语义”的强特征入口（比 count-proxy 更干净）

### 3.2 Meta JSON（每条轨迹一个 json）

字段（README + 样例确认）：
- `Filename`, `Uploaded`, `Points`
- `Start coordinate`, `End coordinate`
- `Owner`, `Description`, `Tags`, `Visibility`（默认不入训练；隐私/偏置风险）
- `Distance`, `Time`
- `geometry`（通常为 [lon, lat] 序列）

建议：
- 训练前先用 Meta 构建全局 manifest（见第 5 节），把“筛选/统计/抽样”从海量小文件 IO 中解耦出来。

---

## 4) Detroit 子集筛选：必须写成可复现规则

WorldTrace 不提供“城市标签”，因此 Detroit 需要用空间范围筛。

### 4.1 空间范围（bbox / polygon）

- 当前采用：两套 bbox 都写入 `docs/DATA_CONTRACT.md`（避免“口径漂移”）：
  - `Detroit official boundary bbox`：来自 ArcGIS `City_of_Detroit_Boundary` extent 3857→4326（用于城市粗筛/审计对照）
  - `Detroit core training window bbox`：用于 `1024×1024` 栅格与训练窗口（用于筛选/落盘/训练）
- 若后续需要更精细边界（polygon，含 enclave/洞），作为扩展消融再加，不影响当前可跑口径。

### 4.2 纳入标准（避免“擦边路过”污染）

当前拍板口径（bbox 粗筛 + bbox 内连续段切片）已写入 `docs/DATA_CONTRACT.md`：
- bbox 内点过滤：按 `Detroit core training window bbox` 过滤出 bbox 内的点
- 连续段切分：若相邻点 `Δt > 5s` 则断开（WorldTrace 标准化后采样间隔为 1s）
- 保留策略：保留 bbox 内**最长连续段**
- 最小段长度：`min_segment_points = 120`（2 分钟，1Hz）
- 段级别质量闸门：该段内 `is_matched=0` 的比例 **> 20%** 则丢弃

---

## 5) 数据形态：文件数/IO 是首要风险（不是训练算力）

WorldTrace 是“海量小文件”（2.45M 条轨迹对应大量 csv/json）。如果直接在训练时逐文件打开，会被 IO 拖死。

### 5.1 建议的中间产物：manifest（强烈建议）

从 `Meta.zip` 萃取一个全局表（建议 parquet/arrow），至少包含：
- `traj_id`（文件名或哈希）
- `traj_path`（csv 路径）
- `start_time/end_time`
- `num_points`
- `bbox`（或 start/end 坐标）
- `distance/duration`（若可用）

Detroit 筛选与抽样只在 manifest 上做，避免反复扫描轨迹文件。

### 5.2 建议的训练落盘格式（两条路二选一）

- 方案 A：`parquet (partitioned)`（按 region/city/date 分区；适合流式读取与筛选）
- 方案 B：`HDF5/LMDB`（你们现有生态更熟；适合连续读取与多 worker）

> 这一步不会“改变科学结论”，但决定你能不能把预训练跑起来。

---

## 6) 把 WorldTrace 用成“预训练底座”：复用 UniTraj 的两策略

UniTraj 的两个核心策略与“是否用地图语义”无关，因此适合作为我们项目的通用增强，并进入消融矩阵。

### 6.1 Adaptive Trajectory Resampling（自适应重采样）

- 目的：轨迹长度跨度很大；长轨迹信息冗余明显、短轨迹信息稀缺。
- 做法：按轨迹长度决定采样率（对数/次线性过渡），压缩长轨迹冗余、保留短轨迹细节。
- 我们的定位：**预训练/微调的数据采样开关**（不改变坐标系与地图语义）。

### 6.2 Self-supervised Masking（多策略遮盖）

- 目的：学习“缺失/断裂下仍保持结构”，提升鲁棒性。
- 四类遮盖（与 UniTraj 一致）：
  - random（模拟随机掉点）
  - block（模拟隧道/峡谷连续缺失）
  - keypoint（RDP 关键拐点遮盖，逼模型学骨架）
  - last-N（对齐预测任务：给前段推后段）
- 我们的定位：**预训练目标**（重建被遮盖点），可迁移到后续任务 backbone。

---

## 7) 与本项目技术路线的对齐（“软约束”哲学）

你已经明确：OSM mask 不应作为训练期 hard support（会切割真实分布、把上限绑定到地图质量）。

因此我们对 WorldTrace 的对齐策略是：
- 训练坐标用 `matched_*`（它已经包含 map matching 的空间一致性）。
- OSM/路网信息只作为：
  - 输入特征（road prior/topology/语义）
  - 审计口径（on-road ratio、corridor error、detour 方向性等）
  - 软正则（必要时）
- 不做：训练期 masked softmax / hard cut。

这与 `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md` 的主线一致（OSM/拓扑/语义作为可消融输入特征）。

---

## 8) 立即要补齐的“数据契约”信息（否则无法开工）

已拍板并写入 `docs/DATA_CONTRACT.md` 的：
- Detroit bbox + 切片/质量闸门口径
- WorldTrace revision + Trajectory/Meta 的 LFS 指纹
- SafeGraph POI：`SAFEGRAPH_PLACES_VINTAGE=2024-01`、`EPSG:4326`、`poi_active(vintage)` 规则

仍需补齐（保持 `TBD`，但必须在开工前写死）的：
- 遥感影像：数据源/时间戳/分辨率/覆盖范围
- 土地利用：数据源/时间戳/类别体系与映射
- OSM：Detroit extract 来源与快照时间（用于软先验/拓扑特征，非硬约束）

---

## 9) 下载（工作站 A 可直接执行的命令）

HuggingFace CLI 下载全量：

```bash
hf download OpenTrace/WorldTrace --repo-type dataset --local-dir <YOUR_DIR>
```

只下载轨迹或元信息：

```bash
hf download OpenTrace/WorldTrace --repo-type dataset --local-dir <YOUR_DIR> --include "Trajectory.zip"
hf download OpenTrace/WorldTrace --repo-type dataset --local-dir <YOUR_DIR> --include "Meta.zip"
```

解压（建议解到两个目录，便于后续批处理与索引）：

```bash
unzip -q Trajectory.zip -d Trajectory
unzip -q Meta.zip -d Meta
```

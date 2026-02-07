# 数据集迁移方案：从 WorldTrace/Detroit 到多模态出租车轨迹

> 日期：2026-02-07
> 目的：为 partner 提供可直接执行的数据下载、预处理、验证方案
> 背景：当前 WorldTrace Detroit 数据集中 83.5% 的 GT 路线 ≈ 最短路径，route generation 退化为图搜索问题，WC 的 latent diversity 无法被验证

---

## 0) 为什么要换数据集

| 问题 | WorldTrace Detroit |
|---|---|
| GT 路线多模态性 | 83.5% detour < 1.2（≈最短路径）|
| 同一 OD 多条路线 | 无（每 OD 仅一条观测）|
| RNN beam=100 Jaccard | p50=1.000（和 GT 完全重合）|
| 含义 | SR/Jaccard 无法区分模型，搜索 trivializes 问题 |

**需要的数据特征**：
1. 同一 OD 区域有大量不同路线（出租车数据天然满足）
2. 路网结构不规则（非棋盘格）
3. 足够的数据量（>10 万条轨迹）
4. 公开可下载、许可证允许研究使用

---

## 1) 推荐数据集排序

### 🥇 首选：Porto Taxi（葡萄牙波尔图）

| 维度 | 说明 |
|---|---|
| 规模 | 442 辆出租车，整年（2013.07 - 2014.06），~170 万条完整行程 |
| 采样率 | **15秒/点**（比 WorldTrace 1Hz 稀疏，但对 way-level 路径匹配够用）|
| 路网 | 欧洲老城区 + 现代道路混合，**非棋盘格**，路线多样性远高于 Detroit |
| 多模态 | 同一出租车站出发、同一区域到达的行程大量存在 → 天然的同 OD 多路线 |
| 获取 | Kaggle 公开下载，1.94 GB 单文件 CSV |
| 许可证 | ECML/PKDD 2015 竞赛数据，学术使用无限制 |
| 被引用 | UniTraj 论文中已用作评估数据集之一 |
| 风险 | 15s 采样 → map matching 精度可能不如 1Hz；需要自行做 map matching |

### 🥈 备选：T-Drive（北京出租车）

| 维度 | 说明 |
|---|---|
| 规模 | 10,357 辆出租车，一周数据，~1500 万点 |
| 采样率 | 不均匀（1s ~ 几分钟），平均 ~2-4 分钟/点 |
| 路网 | 北京环路 + 胡同，路网复杂度高 |
| 多模态 | 有，但采样率太稀疏 → map matching 困难 |
| 获取 | Microsoft Research 公开，OneDrive 下载 |
| 许可证 | 学术使用 |
| 风险 | **采样率太低**，map matching 到 way-level 精度存疑 |

### ❌ 不可用：DiDi GAIA（成都/西安）

- 官方下载页面已关闭（outreach.didichuxing.com 已失效）
- UniTraj 论文也标注为 "proprietary / largely inaccessible"
- 放弃

### ❌ 不推荐：GeoLife

- 178 用户，轨迹量太少（~17K 条），多为步行/骑行
- 不适合车辆路径生成

---

## 2) Porto Taxi 下载与预处理方案

### 2.1 下载

```bash
# 方式 A：Kaggle CLI（推荐：优先使用竞赛源；失败再 fallback dataset 源）
#
# 0) 约定：工作站数据根目录（不要写死 /data）
export RAW_ROOT="${RAW_ROOT:-$HOME/data/geoexplicit_data}"
mkdir -p "$RAW_ROOT/porto_taxi/raw"

# 1) 安装/配置 Kaggle CLI
pip install kaggle
# 把 kaggle.json 放到 ~/.kaggle/kaggle.json 并 chmod 600

# 2) 优先：竞赛源（包含 train/test + taxi stands metadata + eval script）
# NOTE: 首次下载前需要先在 Kaggle 页面点击 "Accept Rules"，否则会 403：
#   https://www.kaggle.com/competitions/pkdd-15-taxi-trip-time-prediction-ii/rules
kaggle competitions download -c pkdd-15-taxi-trip-time-prediction-ii -p "$RAW_ROOT/porto_taxi/raw"
unzip -o "$RAW_ROOT/porto_taxi/raw/"*.zip -d "$RAW_ROOT/porto_taxi/raw"

# 3) fallback：dataset mirror（只有 train.csv；license 标注可能不清晰）
# kaggle datasets download -d crailtap/taxi-trajectory -p "$RAW_ROOT/porto_taxi/raw"
# unzip -o "$RAW_ROOT/porto_taxi/raw/"*.zip -d "$RAW_ROOT/porto_taxi/raw"

# 4) 解压成功后删除 zip（节省磁盘；删前做 sanity check）
head -n 1 "$RAW_ROOT/porto_taxi/raw/train.csv"
wc -l "$RAW_ROOT/porto_taxi/raw/train.csv"
rm -f "$RAW_ROOT/porto_taxi/raw/"*.zip

# 方式 B：浏览器下载
# https://www.kaggle.com/datasets/crailtap/taxi-trajectory
# 下载 train.csv（1.94 GB）
```

> 推荐直接运行脚本（已在仓库提供，避免口径漂移）：
>
> ```bash
> RAW_ROOT=/home/jinlin/data/geoexplicit_data bash tools/porto/porto_download_raw.sh
> ```

### 2.2 数据格式

单文件 `train.csv`，每行一条完整行程：

| 字段 | 类型 | 说明 |
|---|---|---|
| TRIP_ID | string | 唯一行程 ID |
| CALL_TYPE | char | A=调度, B=出租车站, C=路边扬招 |
| ORIGIN_CALL | int | 电话 ID（CALL_TYPE=A 时有值）|
| ORIGIN_STAND | int | 出租车站 ID（CALL_TYPE=B 时有值）|
| TAXI_ID | int | 出租车 ID |
| TIMESTAMP | int | Unix 时间戳（秒）|
| DAYTYPE | char | A=普通日, B=假日, C=假日前一天 |
| MISSING_DATA | bool | GPS 流是否完整 |
| POLYLINE | string | GPS 坐标序列，JSON 格式 `[[lon,lat], ...]`，**每 15 秒一个点** |

### 2.3 预处理 Pipeline（partner 执行）

```
Step 1: 基础清洗
├── 过滤 MISSING_DATA=True 的行程
├── 解析 POLYLINE JSON → numpy array (N, 2) [lon, lat]
├── 过滤点数 < 10 的行程（< 150 秒，太短）
├── 过滤点数 > 300 的行程（> 75 分钟，异常）
└── 输出：cleaned_trips.parquet（trip_id, taxi_id, timestamp, polyline, n_points）

Step 2: OSM 路网准备
├── 下载 Porto 区域的 OSM 数据（Portugal extract from Geofabrik）
│   URL: https://download.geofabrik.de/europe/portugal-latest.osm.pbf
│   保存到：$RAW_ROOT/osm/portugal-latest.osm.pbf
├── （可选但推荐）用 osmium 裁剪 Porto bbox（秒级）
│   bbox 约：-8.72,41.10,-8.52,41.22
│   输出：$RAW_ROOT/osm/porto_extract.osm.pbf
├── 用 osmnx 或自建脚本构建 Porto 的 way-level 有向图
│   bbox: 大约 [-8.72, 41.10, -8.52, 41.22]（波尔图市区）
├── 输出：road_graph.npz（与当前 Detroit 格式一致）
│   包含：way_adj_ptr, way_adj_idx, way_center, way_len_m, way_tier, ...
└── 记录：n_ways, avg_degree

Step 3: Map Matching
├── 工具推荐：Valhalla（推荐）或 OSRM
│   Valhalla: docker pull ghcr.io/valhalla/valhalla:latest
│   优势：原生支持 15s 间隔的稀疏轨迹 matching
├── 输入：每条轨迹的 GPS 点序列
├── 输出：每条轨迹的 way_id 序列（ordered list of OSM way IDs）
├── 质量闸门：matching confidence < 阈值 → 丢弃
└── 输出：matched_routes.parquet（trip_id, way_sequence, match_confidence）

Step 4: 构建训练数据（与当前 pipeline 对齐）
├── 将 way_sequence 映射到 road_graph.npz 的内部 way_id
├── 过滤：way_sequence 长度 >= 5 hops（min_hops=5 与当前一致）
├── 统计：每个 OD region-pair 的路线数量
├── 构建 OD-disjoint split（与当前协议一致）
└── 输出格式与当前 Detroit 的 graph_routes.npz 一致
```

### 2.4 关键验证指标（预处理完成后立即检查）

```python
# 在 matched_routes 上运行：
1. n_trips_total, n_trips_matched, match_rate
2. n_ways, avg_degree  # 确认路网复杂度
3. way_sequence 长度分布（p25, p50, p75, p95）
4. Shortest path detour ratio = GT_length / SP_length
   → 目标：p50 > 1.05, detour > 1.2 的比例 > 25%
5. 同 OD-region 的路线数量分布
   → 目标：至少 30% 的 OD-region pair 有 >= 3 条不同路线
```

**如果 detour ratio 依然 ≈ 1.0、同 OD 路线高度重合，说明出租车也走最短路径——问题不在数据集而在任务本身。这个结果也是有价值的。**

---

## 3) 工程对接：需要改动的模块

### 3.1 不需要改的（架构层）
- WC 模型代码（way_decoder.py, way_encoder.py）
- RNN baseline（rnn_ar.py）
- 训练循环（training/）
- 评估框架（evaluation/way_casd_binned_eval.py）

### 3.2 需要改的（数据层）
- `src/data/road_graph/` 下的图构建脚本 → 适配 Porto OSM
- 图路径提取：`dump_graph_paths_from_routes_npz.py` → 适配新的 matched 轨迹格式
- Region 划分：Louvain clustering → 在 Porto 图上重新跑
- way_encoder 的几何特征（coord, direction, highway_type）→ 自动适配新图，无代码改动
- **RNN 的 way embedding 维度**：`nn.Embedding(n_ways, d_model)` → n_ways 变化，需重新训练
- 配置文件：bbox, n_ways, 各种路径常量

### 3.3 新增的（map matching）
- Map matching pipeline（Valhalla Docker）
- 轨迹清洗脚本
- 质量审计脚本

---

## 4) 时间估算

| 步骤 | 耗时估计 | 阻塞项 |
|---|---|---|
| 下载 Porto CSV + OSM | 1 小时 | 网络带宽 |
| 清洗 + 解析 | 2 小时 | 无 |
| OSM 图构建 | 3 小时 | osmnx / 自建脚本 |
| Valhalla 部署 + Map matching | **1-2 天** | Docker + 170 万条轨迹 matching |
| 数据格式转换 + Split | 半天 | 无 |
| Region clustering | 2 小时 | 无 |
| 训练 WC + RNN（各 1 天） | 2 天 | GPU |
| 评估 + 对比 | 半天 | 无 |
| **总计** | **~5 天** | Map matching 是瓶颈 |

---

## 5) 风险与备选

| 风险 | 影响 | 缓解 |
|---|---|---|
| 15s 采样 → map matching 精度低 | way sequence 不准确 | 用 Valhalla 的 trace_route API，它专门处理稀疏轨迹 |
| Porto 路网也以最短路径为主 | 同 Detroit 一样的问题 | Step 2.4 的验证指标会尽早暴露；如果真如此，转向 T-Drive (北京) |
| Valhalla 部署复杂 | 拖慢进度 | 备选方案：OSRM match API + Docker |
| n_ways >> Detroit (19K) | 训练变慢 | WC 不用 way embedding (用几何特征)，不受影响；RNN 需要更大 embedding |

---

## 6) Partner 执行清单

```
□ 1. 下载 Porto taxi CSV（Kaggle）
□ 2. 下载 Portugal OSM（Geofabrik）
□ 3. 部署 Valhalla Docker
□ 4. 运行清洗脚本（过滤 + 解析 POLYLINE）
□ 5. 构建 Porto way-level 有向图（road_graph.npz 格式）
□ 6. 运行 map matching（Valhalla trace_route）
□ 7. 统计 detour ratio + 同 OD 路线多样性 → 报告数字
□ 8. 如果通过验证 → 构建 graph_routes.npz + OD-disjoint split
□ 9. 运行 Louvain region clustering
□ 10. 训练 WC AE + Region AR + Flow → 评估
□ 11. 训练 RNN baseline → 评估
□ 12. 对比结果
```

**关键检查点：Step 7 完成后暂停**，报告 detour ratio 和路线多样性数字。如果数据仍缺乏多模态性，在投入训练之前重新评估方向。

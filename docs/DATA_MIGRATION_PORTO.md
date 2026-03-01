# 数据集迁移方案：从 WorldTrace/Detroit 到多模态出租车轨迹

> 日期：2026-02-07（更新 2026-02-08：优化执行流程，删除冗余步骤）
> 目的：为 partner 提供可直接执行的数据下载、预处理、验证方案
> 背景：当前 WorldTrace Detroit 数据集中 83.5% 的 GT 路线 ≈ 最短路径，route generation 退化为图搜索问题，WC 的 latent diversity 无法被验证

## ⚡ 快速执行指南

**核心思路：只新写一个 CSV→parquet 转换脚本，后续完全复用现有 `scripts/data_prep/run_way_casd_prep.sh` pipeline。**

### ✅ 当前进度（Porto 已跑通，产物已落盘）

本次在工作站完成的 Porto 数据预处理输出根目录：

- `OUT_BASE=/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0`

关键产物（Way-CASD 标准接口，W1–W4）：

- `W1_way_routes/way_routes.npz`
- `W2_way_graph/way_graph.npz`
- `W3_way_features/way_features.npz`
- `W4_way_routes_labeled/way_routes_labeled.npz`（推荐后续训练/评测用这个）

路由与标签统计（来自 `W4_way_routes_labeled/way_routes_labeled.npz`）：

- `N=1,630,527`（routes），`vocab=35,471`（ways）
- `way_seq_len`：p50=24，p90=46，mean=27.3，max=212
- `corridor_type`：
  - 0：1,378,968
  - 1：213,802
  - 2：1,295
  - 3：36,462

### ✅ 质量诊断结果（A_porto_diagnose）

诊断输出目录：
- `A_porto_diagnose=/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0/A_porto_diagnose`

Way graph 审计（`W2_way_graph/way_graph.npz`）：
- `ways=35,471`，`edges_directed=280,830`
- `out_deg`：p50=4，p90=18，max=211
- 连通性：`n_cc=1`，`largest_cc=100%`，`isolated=0%`

Way routes 质量审计（`W4_way_routes_labeled/way_routes_labeled.npz`，仅统计 `len∈[3,160]`）：
- `N=1,629,126`（routes）
- `way_seq_len`：p50=24，p90=46，max=160
- `loop_ratio`：p50=0.000，p90=0.026，p99=0.138
- `missing_frac`：p50=0.000，p90=0.047，p99=0.200
- `dead_end_frac`：p50=0.000，p90=0.000，max=0.000
- `max_step_m`：p50=494m，p95=14,899m，p99=15,436m，max=25,149m（用于定位异常跳跃/超长 way；详见 `way_routes_bad.json`）

Shortest path baseline（detour ratio）：
- 已对 `n=500` routes 计算并保存：`A_porto_diagnose/shortest_path_baseline.json`
- 下一步必须读取该文件确认 **GT 是否明显偏离最短路**（否则 Porto 也会退化成“图搜索/最短路”问题）

OD bin 路径多样性扫描（corridor diversity）：
- 产物：`A_porto_diagnose/od_diversity_scan.json`（从 coarse OD bin 内的多条 way 序列计算 LCS distance）
- 统计（一次采样分析 `5000` 个 OD bins，阈值 `max_pairwise_LCS_dist>=0.5` 判为 multimodal）：
  - `unique OD bins=20,488`，其中 `>=5 routes` 的 valid bins=8,779
  - routes per bin（valid）：p50=23，p90=344，p99=2,599，max=18,836
  - multimodal bins：`4,980/5,000 = 99.6%`
  - mean LCS distance（bin 内 pairwise 平均）：mean=0.742，p50=0.763，p90=0.871
- 解释：Porto 在**粗粒度 OD 口径**下存在非常显著的 corridor 多样性（这正是 Way-CASD / Flow latent diversity 能发挥作用的前提）。
- ⚠️ 注意：当前 `max_step_m` 尾部很重（p95≈14.9km），说明仍存在“way 间跳跃/teleport edge”风险；建议在用 detour/SP 或更严格指标前先做异常过滤或用 OSM topology 重建 graph。

### ✅ 下一步（P0 / Blocking）：Strict gate + OD-disjoint split

PI 建议的 strict gate 不是为了过滤 detour（那是多样性信号），而是过滤**数据质量问题**（teleport / missing / 异常循环）。本仓库已在 `A_porto_diagnose/way_routes_bad.json` 中生成默认阈值下的 bad route 列表（`n_bad=278,983`），对应保留：

- keep `1,350,143 / 1,629,126 = 82.9%`（在 `len∈[3,160]` 子集内）

建议直接用脚本一键生成 strict 数据集 + split（小文件可进 git，npz 不进）：

```bash
export RAW_ROOT="$HOME/data/geoexplicit_data"
bash tools/porto/run_porto_strict_gate_and_split.sh
```

输出：
- `W5_way_routes_strict_gate/way_routes_strict_gate.npz`（strict routes；包含 `orig_route_id` 映射）
- `W5_way_routes_strict_gate/od_split_min3_max160_seed0.json`（OD-disjoint split，供训练/评测复用）
- `W5_way_routes_strict_gate/report.json`（过滤统计）

### 前置条件
1. `train.csv` 已存在于 `$RAW_ROOT/porto_taxi/raw/`（✅ 已完成）
2. `portugal-latest.osm.pbf` 已存在于 `$RAW_ROOT/osm/`（✅ 已完成，382MB）
3. Valhalla Docker 在 `localhost:8002` 运行（需部署，见下方）

### 一键执行
```bash
export RAW_ROOT="$HOME/data/geoexplicit_data"

# 部署 Valhalla (首次，约 20 分钟)
# 详见下方 "Valhalla 部署" 一节

# 调试跑 100 条，确认 Valhalla 通路正常
bash tools/porto/run_porto_prep.sh --limit 100

# 全量跑 (Phase 0: ~2-3h map matching; Phase 1: ~15min pipeline)
bash tools/porto/run_porto_prep.sh
```

### 代码结构
```
tools/porto/
  porto_bbox_meta.json              ← Porto bbox+grid 定义
  porto_csv_to_segments_parquet.py  ← 唯一新代码: CSV→parquet (Valhalla)
  run_porto_prep.sh                 ← 入口: Phase 0 + 复用 scripts/data_prep/run_way_casd_prep.sh
  run_porto_diagnose.sh             ← 诊断：图/质量/最短路 baseline
  run_porto_strict_gate_and_split.sh← P0：strict gate + OD-disjoint split
  porto_od_diversity_scan.py        ← OD bin corridor 多样性扫描
  porto_download_raw.sh             ← 数据下载 (已完成)
```

### 计算流程
```
Porto train.csv                     WorldTrace Trajectory.zip
     │                                    │
     ▼  porto_csv_to_segments_parquet.py  ▼  build_detroit_segments.py
  segments_with_wayid.parquet  ←  同一格式  →  segments_with_wayid.parquet
     │                                    │
     └──────────── 完全相同的 pipeline ───────────┘
                        │
                        ▼  scripts/data_prep/run_way_casd_prep.sh
                   W1: way_routes.npz
                   W2: way_graph.npz
                   W3: way_features.npz
                   W4: way_routes_labeled.npz
```

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

### 2.3 预处理 Pipeline

**核心洞见：** 现有 `scripts/data_prep/run_way_casd_prep.sh`（4 步）完全可复用。唯一新代码是 Phase 0：Porto CSV → `segments_with_wayid.parquet`。

#### Phase 0: CSV → segments_with_wayid.parquet（唯一新代码）

脚本：`tools/porto/porto_csv_to_segments_parquet.py`

```
输入: train.csv (1.7M rows)
  ├── 过滤 MISSING_DATA=True
  ├── 解析 POLYLINE JSON → (N,2) [lon,lat]
  ├── 过滤点数 ∉ [10, 300]
  ├── Valhalla trace_attributes API → 每个 GPS 点的 osm_way_id
  │     ├── edges[].way_id + matched_points[].edge_index
  │     ├── gps_accuracy=30, search_radius=50（适应 15s 稀疏采样）
  │     └── unmatched 点 → osm_way_id = 0
  └── 输出: segments_with_wayid.parquet（同 WorldTrace 格式，12 列）
```

**Parquet Schema（与 WorldTrace 完全一致）：**
```
traj_csv: string, n_points: int32, unmatched_ratio: float32,
way_id_missing_ratio: float32, t: list<int64>, lat: list<float32>,
lon: list<float32>, y: list<int32>, x: list<int32>,
is_matched: list<int8>, matched_distance: list<float32>,
osm_way_id: list<int64>
```

#### Phase 1: 现有 pipeline（零代码改动）

由 `scripts/data_prep/run_way_casd_prep.sh` 执行，通过环境变量注入 Porto 路径：

```
W1: build_way_routes_from_segments_parquet.py → way_routes.npz
    (读 osm_way_id → 去重连续重复 → 构建路线)
W2: build_way_graph_from_way_routes_npz.py → way_graph.npz
    (从路线构建 CSR 邻接图)
W3: build_way_features_from_osm_pbf.py → way_features.npz
    (pyrosm 从 PBF 提取 highway_type/geometry/length)
W4: label_corridor_type_from_way_features.py → way_routes_labeled.npz
    (基于 way_features 标注 corridor_type)
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
- 配置文件：bbox, n_ways, 各种路径常量 → 由 `run_porto_prep.sh` 通过环境变量注入
- `porto_bbox_meta.json` → 复制为 `semantic/osm_road_prob_meta.json`（供 `build_way_features_from_osm_pbf.py` 读取 bbox）
- **RNN 的 way embedding 维度**：`nn.Embedding(n_ways, d_model)` → n_ways 变化，需重新训练

### 3.3 新增的（已完成）
- `tools/porto/porto_csv_to_segments_parquet.py` — CSV→parquet 转换（含 Valhalla map matching）
- `tools/porto/porto_bbox_meta.json` — Porto bbox+grid 元数据
- `tools/porto/run_porto_prep.sh` — 入口脚本（Phase 0 + Phase 1）
- Valhalla Docker 部署（一次性，见 run_porto_prep.sh 注释）

---

## 4) 时间估算

| 步骤 | 耗时估计 | 阻塞项 |
|---|---|---|
| 下载 Porto CSV + OSM | ✅ 已完成 | — |
| Valhalla Docker 部署 | ~20 分钟 | osmium 裁剪 + tile 构建 |
| Phase 0: CSV→parquet (map matching) | **2-3 小时** | 8 workers 并行 |
| Phase 1: way_routes→graph→features | ~15 分钟 | 无 |
| 验证检查点（detour + diversity） | 30 分钟 | 人工审核 |
| 训练 WC + RNN（各 1 天） | 2 天 | GPU |
| 评估 + 对比 | 半天 | 无 |
| **总计** | **~3 天** | Phase 0 map matching 是瓶颈 |

---

## 5) 风险与备选

| 风险 | 影响 | 缓解 |
|---|---|---|
| 15s 采样 → map matching 精度低 | way sequence 不准确 | 用 Valhalla 的 `trace_attributes` API（`gps_accuracy=30, search_radius=50`），专门处理稀疏轨迹 |
| Porto 路网也以最短路径为主 | 同 Detroit 一样的问题 | Phase 1 完成后立即检查 detour ratio；如果仍 ≈1.0，转向 T-Drive (北京) |
| Valhalla 部署复杂 | 拖慢进度 | 备选方案：OSRM match API + Docker |
| n_ways >> Detroit (19K) | 训练变慢 | WC 不用 way embedding (用几何特征)，不受影响；RNN 需要更大 embedding |

---

## 6) 执行清单

```
✅ 1. 下载 Porto taxi CSV（Kaggle）  — 已完成，1,710,671 rows
✅ 2. 下载 Portugal OSM（Geofabrik）  — 已完成，382MB
✅ 3. Phase 0+1 全量跑通（Porto way 数据已产出 W1–W4）
□  4. 🔴 验证检查点：detour ratio + 同 OD 路线多样性 → 报告数字
□  5. 如果通过验证 → 训练 WC + RNN → 评估 → 对比
```

**关键检查点：Step 4 完成后暂停**，报告 detour ratio 和路线多样性数字。如果数据仍缺乏多模态性，在投入训练之前重新评估方向。

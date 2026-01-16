# Way-CASD 方法说明（核心口径）

> 本文档回答三个核心问题：  
> 1) Way-CASD 用的“路线 token”是什么；  
> 2) Way-CASD 模型的训练/采样闭环是什么；  
> 3) 我们如何在 GT 数据里定义/检测 “corridor-level 多模态”（用于数据筛选与评估）。

---

## 1. 为什么必须用 way token（而不是 node/segment）

WorldTrace 的每个点提供 `osm_way_id`（点匹配到的 OSM way）。  
在 1Hz 轨迹上，如果先把 GPS 点 snap 到 node，再做最短路 bridging，常见会产生 **千级 node 序列**，使任何 AR/扩散都被“长度与累积误差”主导。

Way-CASD 的关键选择是：

- **路线表示**：`route = [way_1, way_2, ..., way_L]`（OSM way id 序列）
- **去重**：只做 **连续重复去重**（consecutive dedup），避免 1Hz 采样在同一条 way 上重复计数
- 目标：把 `L` 压到几十步（对齐 GTG/Cardiff 的粒度）

对应代码：

- 抽取含 `osm_way_id` 的 segments parquet：`src/data/worldtrace/build_detroit_segments.py`
- way 序列长度审计：`src/data/worldtrace/way_seq_stats_from_segments.py`

---

## 2. Way-CASD：两步训练（AE → Flow）

### 2.1 输入与数据对象

Way-CASD 的最小训练数据是：

- `way_routes*.npz`：每条路线的 way 序列（CSR）+ 条件（OD/time/city）
- `way_graph*.npz`：way-level 邻接（CSR，来自 GT transitions；可选做 undirected）
- `way_features*.npz`：每个 way 的特征（长度/中心/方向/road tier/highway type）

数据格式详见：`docs/DATA_STRUCTURE.md` 的 Way routes / Way graph 小节。

### 2.2 Step A：Way 序列自编码（AutoEncoder）

目的：先验证 **编码→解码** 能重建 GT transitions，避免 Flow 在“不稳定 latent”上学习。

- WayEncoder（无 ID lookup）  
  输入：`way_id` → 输出：`(center_yx, dir_yx, log1p(len_m), tier, highway_code)` 的特征嵌入  
  代码：`src/models/way_casd/way_encoder.py`
- Perceiver 压缩  
  变长 token → 固定长度 latent tokens  
  代码：`src/models/casd/perceiver.py`（复用）
- Constrained AR Decoder（候选集打分）  
  每步只在 `succ(way)` 候选集中预测下一跳；beam search 可提升到达率  
  代码：`src/models/way_casd/way_decoder.py`
- 条件向量（ConditionEncoder）  
  `start_pos/dest_pos + hour/dow + route_city (+ corridor_type 可选)`  
  代码：`src/models/way_casd/conditions.py`

训练入口：`src/training/train_way_casd_autoencoder.py`

### 2.3 Step B：latent Flow Matching（生成核心）

目的：学习 `p(z | condition)`，采样 latent 后再解码得到 way 序列。

- Flow：rectified flow / flow matching in latent token space  
  代码：`src/models/way_casd/latent_flow.py`
- 采样：ODE 迭代 +（可选）CFG（对 corridor_type 做 dropout → guidance）

训练入口：`src/training/train_way_casd_flow.py`

### 2.4 可视化（最小闭环证据）

- Way-CASD 采样可视化：`src/evaluation/way_casd_sample_viz.py`  
  关键看：到达率、长度是否“撞墙”（`len == max_decode_len`）、以及与 GT 的 overlap/Jaccard。

---

## 3. corridor 是什么（以及它不是啥）

我们区分两个概念（避免讨论混淆）：

1) **corridor（走廊 / mode）**：同一 OD（或 OD-bin）下，GT 路线在空间上形成的多个稳定走廊分布。  
   - 这是“数据/行为”概念，用于 **数据筛选与评估**。
2) **corridor_type（4 类标签）**：我们为了做最小可控性（CFG）引入的 **粗粒度 route label**（dominant tier）。  
   - 这是“工程条件变量”，不是 corridor 本体；它不应该替代 corridor 的空间分布定义。

`corridor_type` 的生成：`src/data/way_graph/label_corridor_type_from_way_features.py`（dominant tier > 0.5 → major/minor/service else mixed）。

---

## 4. corridor-level 多模态扫描（GT 证据的自动化口径）

### 4.1 扫描目标

从 `Trajectory.zip` 中筛出 “同 OD-bin 有多条 GT 且存在多走廊” 的 OD bins，用于：

- 数据筛选（只取多模态 OD 的路线训练/评估）
- PI sanity（可视化 top-K OD 的走廊是否合理）

入口脚本：`src/data/worldtrace/scan_multimodal_od_region.py`

### 4.2 扫描逻辑（核心计算）

对每条轨迹（zip member）：

1) bbox gate：要求轨迹点落入 bbox 的比例 ≥ `min_points_in_bbox_ratio`
2) OD gate：`od_km >= min_od_dist_km`
3) OD bin：  
   `o_lon_bin=floor(o_lon/od_bin_deg)`，`o_lat_bin=floor(o_lat/od_bin_deg)`，同理得到 `d_*`  
   `od_bin=(o_lon_bin,o_lat_bin,d_lon_bin,d_lat_bin)`
4) route signature（用于聚类）：
   - 直接取 `osm_way_id`（连续去重）作为 signature：`sig = (w1,w2,...,wL)`  
   - 截断：`L <= max_way_seq_len`（避免极端长序列拖垮聚类）
   - OD 的起讫点用 **第一个/最后一个出现有效 way_id 的点坐标**（而不是首末 GPS 点），保证 `od_bin` 与 `signature` 对齐。

对每个 `od_bin`：

- 得到若干 signature 的计数（最多保留 top-`max_sigs_per_od`）
- 先做“近似去重/合并”（greedy merge）：若两条 signature 的距离 < `merge_dist_thr` 则合并计数
- 判断是否 multimodal：
  - `n_routes >= min_routes_per_od`
  - 第二大簇比例 ≥ `min_cluster_frac`
  - top2 separation ≥ `cluster_sep_thr`

距离度量（当前版）：

- **LCS distance**：  
  `dist(a,b) = 1 - LCS(a,b) / max(len(a),len(b))`，范围 `[0,1]`

### 4.3 可视化 sanity（corridor footprint）

用于 PI 快速判断“检测到的 OD 是否真的多走廊”：

- 脚本：`src/evaluation/plot_worldtrace_multimodal_od_bins.py`
- 左图：两簇代表轨迹（经 downsample）+ 灰色路网背景（可选）
- 右图：用 `route_bin_deg` 把轨迹点离散到 coarse grid，画两簇 footprint + overlap

注意：这里的 footprint 只是为了“肉眼可解释”；multimodal 的判定口径以 scan 脚本为准。

#### 4.3.1 可视化加速：先导出 viz cache（推荐）

问题：`Trajectory.zip` 里有百万级 CSV，直接在可视化脚本里逐个随机读取会很慢（而且容易因为 IO 抖动导致耗时不稳定）。

解决：扫描完成后，把每个 multimodal OD 的代表轨迹（每簇 1–3 条）先抽出来存为一个小的 `viz_cache.npz`，后续画图不再读 zip。

- 导出缓存：`src/data/worldtrace/dump_multimodal_viz_cache.py`
- 可视化读取缓存：`src/evaluation/plot_worldtrace_multimodal_od_bins.py --viz_cache_npz ...`

典型工作站命令（48 workers）：

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen/A_mm_od_mioh_v2_bin02_sep50"

# 1) dump cache（top200 OD，每簇最多2条代表轨迹）
python -m src.data.worldtrace.dump_multimodal_viz_cache \
  --scan_report_json "$EXP_ROOT/report.json" \
  --trajectory_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Trajectory.zip" \
  --out_npz "$EXP_ROOT/viz_cache_top200.npz" \
  --top_k 200 --clusters_keep 2 --max_files_per_cluster 2 \
  --prefer_matched --downsample_step 10 \
  --num_workers 48 --chunk_size 256 --mp_start fork

# 2) 随机画 5 个 OD（不再读 zip）
python -m src.evaluation.plot_worldtrace_multimodal_od_bins \
  --scan_report_json "$EXP_ROOT/report.json" \
  --viz_cache_npz "$EXP_ROOT/viz_cache_top200.npz" \
  --out_dir "$EXP_ROOT/viz_rand5_seed0" \
  --random_k 5 --seed 0 \
  --prefer_matched --downsample_step 10 \
  --osm_pbf_michigan "$RAW_ROOT/osm/michigan-latest.osm.pbf" \
  --osm_pbf_ohio "$RAW_ROOT/osm/ohio-latest.osm.pbf"
```

Implementation Plan, Task List and Thought in Chinese

# 开发进度追踪（当前主线：WorldTrace × Detroit）

> [!IMPORTANT]
> **任务口径/评估口径**以 `docs/TASK_DEFINITION.md` 为唯一准则；本文件只记录“主线推进到哪一步、下一步需要产出什么”。  
> legacy（深圳 dt30）已归档：`docs/archive/legacy_shenzhen/`。

---

## 0) 主线叙事（1 句话）

用 **Behavioral Reference Frame**（从 functional 城市学到的“正常 route choice”）去预测 Detroit 的“正常该怎么走”，把预测与真实的差做空间化，得到 **Behavioral Avoidance Field** 并与外部断裂指标对齐（H1/H2/H3）。

---

## 1) 代码能力就绪情况（不是“结果”）

| 模块 | 目标产物 | 入口 |
|---|---|---|
| WorldTrace manifest | `meta_manifest.parquet` | `python -m src.data.worldtrace.build_manifest` |
| WorldTrace Detroit segments | `segments.parquet` | `python -m src.data.worldtrace.build_detroit_segments` |
| OSM soft prior | `osm_road_mask.npy / osm_dist_to_road_m.npy / osm_road_prob.npy` | `python -m src.data.osm.build_osm_road_prob` |
| SafeGraph POI rasters | `poi_density_*.npy / landuse_dom.npy / landuse_entropy.npy` | `python -m src.data.safegraph.build_poi_rasters` |
| Wayback imagery | `wayback tiles (jpg)` | `python -m src.data.wayback.download_wayback_tiles` |
| Census/ACS external indicators | `acs_tract.csv + tiger tracts` | `python -m src.data.census.download_acs_tract` / `python -m src.data.census.download_tiger_tract` |

---

## 2) Phase D（Detroit）最小可跑流程（只写“下一步需要产出什么”）

> 路径与版本约束见 `docs/DATA_CONTRACT.md`；Wayback 的代理/SSL 口径见 `docs/WAYBACK.md`。

### Step D1：构建 WorldTrace manifest（全量索引，解耦海量小文件 IO）

```bash
python -m src.data.worldtrace.build_manifest \
  --meta_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Meta.zip" \
  --out_manifest "$RAW_ROOT/worldtrace/meta_manifest.parquet"
```

### Step D2：筛 Detroit core segments（bbox 内最长连续段）

```bash
python -m src.data.worldtrace.build_detroit_segments \
  --trajectory_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Trajectory.zip" \
  --out_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments.parquet" \
  --dt_gap_s 5 --min_segment_points 120 \
  --matched_distance_max_m 30 --max_unmatched_ratio 0.2 \
  --num_workers 48 --chunk_size 5000
```

### Step D3：生成 OSM road_prob（soft prior）

```bash
python -m src.data.osm.build_osm_road_prob \
  --osm_pbf "$RAW_ROOT/osm/michigan-latest.osm.pbf" \
  --out_dir "$RAW_ROOT/worldtrace/detroit_core_v1" \
  --road_types B --buffer_m 15 --road_prob_sigma_m 50
```

### Step D4：生成 POI/功能区栅格（SafeGraph）

```bash
python -m src.data.safegraph.build_poi_rasters \
  --base_dir "$RAW_ROOT/safegraph/safegraph_unzip" \
  --base_glob "*.csv" \
  --vintage 2024-01 \
  --out_dir "$RAW_ROOT/worldtrace/detroit_core_v1"
```

### Step D5：下载 Wayback 影像（多 release；注意 proxy）

```bash
PYTHONUNBUFFERED=1 \
HTTP_PROXY="http://127.0.0.1:7890" HTTPS_PROXY="http://127.0.0.1:7890" \
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$RAW_ROOT/wayback/detroit_core_z16_fixed_multi_r6" \
  --bbox -83.25 42.25 -82.95 42.50 \
  --zoom 16 --max_threads 16 \
  --mode fixed_releases \
  --release_ids 10 4756 9175 16245 27946 64776 \
  > >(tee "$RAW_ROOT/wayback/detroit_core_z16_fixed_multi_r6/cli_stdout.json") \
  2> >(tee "$RAW_ROOT/wayback/detroit_core_z16_fixed_multi_r6/cli_stderr.log" >&2)
```

### Step D6：下载 tract-level 外部指标（ACS + TIGER）

```bash
python -m src.data.census.download_acs_tract \
  --year 2023 --state_fips 26 \
  --out_csv "$RAW_ROOT/census/acs_tract_2023_state26.csv"

python -m src.data.census.download_tiger_tract \
  --year 2023 --state_fips 26 \
  --out_dir "$RAW_ROOT/census/tiger_tract_2023_state26" \
  --convert_geoparquet
```


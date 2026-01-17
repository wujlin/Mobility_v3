# WorldTrace Phase 2：空间可视化（Detroit｜Layer 1-3）

## Thought（结论先行）

这套可视化不是“讲故事”，而是把三个递进问题钉死：

1) **数据覆盖了哪里？**（Layer 1：轨迹热力图）  
2) **出行从哪到哪？**（Layer 2：O/D 空间分布）  
3) **同一 owner 的同一 OD 是否真的存在多走廊？**（Layer 3：within-owner corridor 聚类 + Top-10 OD 出图）

> 重要前提：Detroit 子集 Owner 只有 2 个且高度集中（见 `docs/WORLDTRACE_OWNER_AUDIT.md`），因此 **between-owner** 分解在 Detroit 上不可行；本阶段只做 **within-owner** 的诊断与可视化。

---

## Layer 1-2：轨迹热力图 + O/D 分布

脚本：`src/evaluation/plot_worldtrace_segments_spatial_layers.py`

输入：
- `segments_with_wayid.parquet`（包含 `y/x/lat/lon/osm_way_id`）
- 可选：`osm_road_prob.npy` 作为灰底 road mapping

输出：
- `detroit_trajectory_heatmap.png`
- `detroit_od_scatter.png`
- `report.json`

工作站命令（Detroit）：

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"

python -m src.evaluation.plot_worldtrace_segments_spatial_layers \
  --segments_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
  --road_prob_npy "$RAW_ROOT/worldtrace/detroit_core_v1/osm_road_prob.npy" \
  --out_dir "$EXP_ROOT/A_spatial_detroit_layers12" \
  --min_od_dist_km 1.0
```

---

## Layer 2.5：Top-1 Owner 性质诊断（时间/距离/时长）

脚本：`src/data/worldtrace/owner_profile_from_segments_with_wayid.py`

输出：`owner_profile_top1.json`

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"

python -m src.data.worldtrace.owner_profile_from_segments_with_wayid \
  --segments_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
  --meta_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Meta.zip" \
  --out_json "$EXP_ROOT/A_owner_profile_detroit/owner_profile_top1.json" \
  --tz_offset_hours -5 \
  --min_od_dist_km 1.0
```

默认只输出 `owner_hash`，不输出 raw owner；如需 raw owner 仅用于内部审计，追加 `--include_owner_raw`。

---

## Layer 3：within-owner corridor（LCS vs Way-level Decision Points）统计 + Top-10 OD 可视化

脚本：`src/data/worldtrace/within_owner_corridor_diversity.py`

输出：
- `corridor_diversity_within_owner.parquet`：每个 OD-bin 的 corridor 统计（**同时包含** LCS 与 decision-point 两套口径）
- `report.json`：Top-10 OD 的 cluster sizes/route_ids（用于出图与 PI sanity）
- `out_viz_dir/top_od_*.png`：Top-10 OD 可视化（颜色按 `--corridor_method` 选择；若为 decision-point 会额外标出 decision points）

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"

python -m src.data.worldtrace.within_owner_corridor_diversity \
  --segments_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
  --meta_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Meta.zip" \
  --road_prob_npy "$RAW_ROOT/worldtrace/detroit_core_v1/osm_road_prob.npy" \
  --out_parquet "$EXP_ROOT/A_within_owner_corridor_detroit/corridor_diversity_within_owner.parquet" \
  --out_json "$EXP_ROOT/A_within_owner_corridor_detroit/report.json" \
  --out_viz_dir "$EXP_ROOT/A_within_owner_corridor_detroit/top_od_viz" \
  --od_bin_deg 0.02 \
  --min_od_dist_km 1.0 \
  --max_way_seq_len 128 \
  --merge_dist_thr 0.15 \
  --corridor_method decision_points \
  --min_choice_count 2 \
  --top_k_od 10
```

解释：
- `od_bin_deg=0.02`：约 2km 粒度，便于聚合出足够 trip 形成可视化走廊。
- `merge_dist_thr`：LCS 距离阈值（用于对照；parquet 中仍会输出 LCS 指标）。
- `corridor_method=decision_points`：Top-10 出图按 **way-level 数据驱动 decision points** 进行 corridor 着色。
- `min_choice_count`：decision-point 去噪阈值；某个 `(way_i -> next_way)` 需在该 OD-bin 内出现 ≥ 该次数，才算“有效选择”。

### 另一路线（推荐）：K-medoids + silhouette 选择 K（避免阈值死循环）

如果我们承认“corridor 的普适定义是 ill-posed”，那就不再试图**发现 K**，而是把 K 作为一个**可解释的选择集规模**：

- 固定候选：`K ∈ {2,3,4}`（route choice 文献常见的“实质不同路径”数量级）
- 距离：仍用 LCS distance（对小变体更宽容）
- 聚类：K-medoids（直接在距离矩阵上工作）
- 选择：silhouette score 最大的 K（簇内紧凑 + 簇间分离）

实现：`within_owner_corridor_diversity.py` 在 Top-OD 里会额外计算 kmedoids，并把 `best_K / silhouette / medoids` 写进 `report.json`。

可视化命令（Top-10 用 kmedoids 着色）：

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"

python -m src.data.worldtrace.within_owner_corridor_diversity \
  --segments_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
  --meta_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Meta.zip" \
  --road_prob_npy "$RAW_ROOT/worldtrace/detroit_core_v1/osm_road_prob.npy" \
  --out_parquet "$EXP_ROOT/A_within_owner_corridor_detroit_km/corridor_diversity_within_owner.parquet" \
  --out_json "$EXP_ROOT/A_within_owner_corridor_detroit_km/report.json" \
  --out_viz_dir "$EXP_ROOT/A_within_owner_corridor_detroit_km/top_od_viz" \
  --od_bin_deg 0.02 \
  --min_od_dist_km 1.0 \
  --max_way_seq_len 128 \
  --merge_dist_thr 0.15 \
  --corridor_method kmedoids \
  --min_choice_count 2 \
  --top_k_od 10
```

如果你需要**全量统计**（回答“Detroit 里到底有多少 OD-bin 是多走廊、对应多少 routes 可用于训练”），追加：

- `--km_eval_all --km_min_routes 5`：对所有 `n_routes>=5` 的 OD-bin 计算 k-medoids，并在 `report.json` 写入 `kmedoids_summary`；同时在输出 parquet 增加 `km_*` 列（bestK/silhouette/second_frac 等）。
- `--viz_random_od 5 --viz_random_seed 0`：额外随机画 5 个 OD（输出到 `top_od_viz/random_od_viz/`），避免只看 Top-10 的“挑样本偏差”。

### decision-point 的关键风险：走廊分叉 vs 走廊内部细节

在 Detroit 的 top-OD 上我们观察到：若把 **所有** decision points 都用于 corridor signature，容易出现 **K 爆炸/大量单例簇**（把“同一走廊内的细节绕行”也当成 corridor）。

因此建议加一个 **主干道路过滤**：只在高等级 road tier 上定义 corridor-defining decision points。

- tier 来自：`way_features.npz` 的 `way_tier`（由 `src/data/way_graph/build_way_features_from_osm_pbf.py` 生成）
- `way_tier` 口径：
  - `0=major`（motorway/trunk/primary/secondary）
  - `1=minor`（tertiary/residential）
  - `2=service`（service/unclassified）
  - `3=other`

过滤参数（在 `within_owner_corridor_diversity.py` 中）：
- `--dp_tier_keep`：逗号/空格分隔的 tier id（如只保留主干路：`--dp_tier_keep 0`）
- `--dp_next_min_keep`：该 decision point 至少要有这么多条“同样在 keep tier 内”的分叉选项（推荐 `2`，用于排除主路/辅路、出口匝道等微分叉）
- 需要同时提供：`--way_features_npz ...`（用于 `osm_way_id -> way_tier` 映射）

推荐命令（Detroit，主干分叉 corridor）：

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"

python -m src.data.worldtrace.within_owner_corridor_diversity \
  --segments_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
  --meta_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Meta.zip" \
  --road_prob_npy "$RAW_ROOT/worldtrace/detroit_core_v1/osm_road_prob.npy" \
  --way_features_npz "$EXP_ROOT/W4_way_features/way_features.npz" \
  --out_parquet "$EXP_ROOT/A_within_owner_corridor_detroit_dp_tier/corridor_diversity_within_owner.parquet" \
  --out_json "$EXP_ROOT/A_within_owner_corridor_detroit_dp_tier/report.json" \
  --out_viz_dir "$EXP_ROOT/A_within_owner_corridor_detroit_dp_tier/top_od_viz" \
  --od_bin_deg 0.02 \
  --min_od_dist_km 1.0 \
  --max_way_seq_len 128 \
  --merge_dist_thr 0.15 \
  --corridor_method decision_points \
  --min_choice_count 2 \
  --dp_tier_keep 0 \
  --dp_next_min_keep 2 \
  --top_k_od 10
```

### decision-point corridor 的口径（核心）

在同一 OD-bin（within 同一 owner）内：

- **Decision Point (way-level)**：某个 `way_i` 的有效 `next_way` 有 ≥2 个（每个选择出现次数 ≥ `min_choice_count`）。
- **Corridor signature**：一条轨迹在这些 decision points 上做出的选择序列 `[(way_i, next_way), ...]`。
- signature 相同的轨迹归为同一 corridor。

这条口径的目标是替代 “整条 way 序列 LCS 聚类” 的过度碎片化：把差异压缩到**少数关键分叉选择**，提升可解释性（“在 way X 选了 Y vs Z”）。

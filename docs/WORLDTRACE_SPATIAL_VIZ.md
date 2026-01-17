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

### decision-point corridor 的口径（核心）

在同一 OD-bin（within 同一 owner）内：

- **Decision Point (way-level)**：某个 `way_i` 的有效 `next_way` 有 ≥2 个（每个选择出现次数 ≥ `min_choice_count`）。
- **Corridor signature**：一条轨迹在这些 decision points 上做出的选择序列 `[(way_i, next_way), ...]`。
- signature 相同的轨迹归为同一 corridor。

这条口径的目标是替代 “整条 way 序列 LCS 聚类” 的过度碎片化：把差异压缩到**少数关键分叉选择**，提升可解释性（“在 way X 选了 Y vs Z”）。

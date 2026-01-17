# WorldTrace Owner 审计（Phase 2 / Step 2.1）

## Thought（结论先行）

本审计回答一个前置问题：**WorldTrace 的 `Owner` 能否作为 panel entity，用于把 corridor diversity 分解为 within-owner vs between-owner？**  
如果 Owner 覆盖不足（同一 OD-bin 内很少出现多个 Owner，或数据高度集中在极少数 Owner），则 Detroit 单城的 panel 分解不可行，需要扩大到 Detroit+Columbus，或更大空间范围。

---

## 事实与字段口径

- `Owner` 位于 **Meta.zip 的 Meta JSON**（不是 Trajectory CSV）。参见 `docs/DATA_CONTRACT.md`。
- Detroit/Columbus 子集的 `segments_with_wayid.parquet` 中包含 `traj_csv`（来自 `Trajectory.zip` 的 member name）。
- 连接键（审计使用的最稳妥口径）：
  - segments：`traj_key = Path(traj_csv).name`
  - meta：`meta_key = Path(Meta.Filename).name`

> 解释：zip 内部可能带目录前缀（例如 `Trajectory/xxx.csv`），用 basename 规避路径不一致。

---

## 需要的统计（Detroit 子集）

1) `unique_owner_count`：Detroit 有多少不同 Owner  
2) `trips_per_owner_distribution`：每个 Owner 的 trips 数分布（p10/p50/p90/max）  
3) `od_coverage_by_owner`（按 `od_bin_deg` 分别统计）：
   - OD-bin 中只有 1 条轨迹的数量（稀疏程度）
   - OD-bin 中出现 ≥2 个不同 Owner 的数量（between-owner 可行性）
   - OD-bin 中只有 1 个 Owner 且该 Owner 在该 OD-bin 中有 ≥2 条轨迹的数量（within-owner 可行性）
4) `owner_type_proxy`：
   - `trips_per_owner > 1000` 的 Owner 数（疑似车队/机构）
   - Owner 的轨迹时间跨度 `< 1 day` 的数量（疑似一次性上传）

---

## Go/No-Go Gate（拍板口径）

- Gate A（between-owner）：
  - 若 `n_od_bins_ge2owners < 100`，则 Detroit 单城的 between-owner 分解不可行，建议扩大到 Detroit+Columbus。
- Gate B（Owner 集中度）：
  - 若 `top10_share >= 0.95`（95% trips 来自 <10 个 Owner），说明数据高度集中，corridor diversity 的解释力受限，需要在结果解释中明确限制，或扩大范围稀释集中度。

---

## 复现命令（工作站）

脚本：`src/data/worldtrace/audit_owner_from_meta_and_segments.py`

### Detroit 单城（推荐先跑，作为 Gate）

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"

python -m src.data.worldtrace.audit_owner_from_meta_and_segments \
  --meta_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Meta.zip" \
  --segments_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
  --out_json "$EXP_ROOT/A_owner_audit_detroit/report.json" \
  --out_md  "$EXP_ROOT/A_owner_audit_detroit/report.md" \
  --od_bin_deg 0.01 0.02 \
  --min_od_dist_km 1.0 \
  --num_top_owners 20
```

### Detroit + Columbus（若 Detroit Gate 失败）

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"

python -m src.data.worldtrace.audit_owner_from_meta_and_segments \
  --meta_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Meta.zip" \
  --segments_parquet \
    "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
    "$RAW_ROOT/worldtrace/columbus_core_v1/segments_with_wayid.parquet" \
  --segments_label detroit columbus \
  --out_json "$EXP_ROOT/A_owner_audit_rustbelt/report.json" \
  --out_md  "$EXP_ROOT/A_owner_audit_rustbelt/report.md" \
  --od_bin_deg 0.01 0.02 \
  --min_od_dist_km 1.0 \
  --num_top_owners 20
```

---

## 输出与隐私

- `report.json`：机器可读的完整统计（不输出原始 Owner，仅输出 `owner_hash` 与计数）
- `report.md`：面向 PI review 的精简摘要（可直接转发）


# WorldTrace Owner 审计（Phase 2 / Step 2.1）

## 结论先行（Go/No-Go）

- `top10_share`（Top-10 Owners 占比）：1.0
- `od_bin_deg=0.01`：`n_od_bins_ge2owners=61`（阈值 100） -> pass=False
- `od_bin_deg=0.02`：`n_od_bins_ge2owners=89`（阈值 100） -> pass=False

## 关键统计

- trips_total=6704, trips_with_owner=6704, owner_found_ratio=1.0
- unique_owner_count=2
- trips_per_owner: p10=828.8000000000001, p50=3352.0, p90=5875.2, max=6506
- owner_time_span_hours: p10=8595.807666666666, p50=11016.164999999999, p90=13436.522333333332, max=14041.611666666666
- n_owner_trips_gt_1000=1, n_owner_span_lt_1day=0

## OD 覆盖（按 od_bin_deg）

- od_bin_deg=0.01: n_od_bins=3733, n_od_bins_1trip=2829, n_od_bins_ge2owners=61, n_od_bins_1owner_ge2trips=843
- od_bin_deg=0.02: n_od_bins=2556, n_od_bins_1trip=1533, n_od_bins_ge2owners=89, n_od_bins_1owner_ge2trips=934

## 可复现命令

建议在工作站 conda 环境（含 `pyarrow`）运行：

```bash
python -m src.data.worldtrace.audit_owner_from_meta_and_segments \
  --meta_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Meta.zip" \
  --segments_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
  --out_json "$EXP_ROOT/A_owner_audit_detroit/report.json" \
  --out_md  "$EXP_ROOT/A_owner_audit_detroit/report.md" \
  --od_bin_deg 0.01 0.02 \
  --min_od_dist_km 1.0 \
  --num_top_owners 20
```

## 备注（隐私）

- 默认只输出 `owner_hash`（sha1 前 8 位）与计数，不输出原始 Owner 字符串。

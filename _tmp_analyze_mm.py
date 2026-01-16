#!/usr/bin/env python3
import json

with open("_sync/wsa/icml2026_routegen/A_mm_od_mioh_v1/report.json") as f:
    d = json.load(f)

s = d["summary"]
print("=== Summary ===")
for k, v in s.items():
    print(f"{k}: {v}")

mm = d["multimodal_od_bins"]
print(f"\n=== Multimodal OD Bins (N={len(mm)}) ===")

n_routes = [x["n_routes"] for x in mm]
print(f"n_routes: min={min(n_routes)}, max={max(n_routes)}, median={sorted(n_routes)[len(n_routes)//2]}")
print(f"Total routes in multimodal ODs: {sum(n_routes)}")

jd = [x["top2_jaccard_dist"] for x in mm]
print(f"jaccard_dist: min={min(jd):.2f}, max={max(jd):.2f}")

print("\n=== Top 10 by n_routes ===")
for i, x in enumerate(mm[:10]):
    od = x["od_bin"]
    lat_o, lon_o = od[1]*0.01, od[0]*0.01
    lat_d, lon_d = od[3]*0.01, od[2]*0.01
    print(f"{i+1}. n={x['n_routes']:2d} clusters={x['cluster_sizes']} J={x['top2_jaccard_dist']:.2f} O=({lat_o:.2f},{lon_o:.2f}) D=({lat_d:.2f},{lon_d:.2f})")

# 计算如果放宽条件的效果
print("\n=== Sensitivity Analysis ===")
print(f"od_bins_with_n_gte_5: {s['od_bins_with_n_gte_5']}")
print(f"od_bins_with_n_gte_10: {s['od_bins_with_n_gte_10']}")

# 检查所有OD的cluster分布
print(f"\n=== Rate Analysis ===")
kept = s["files_kept_after_filter"]
mm_routes = sum(n_routes)
print(f"Kept trajectories: {kept}")
print(f"Multimodal routes: {mm_routes}")
print(f"Multimodal rate: {mm_routes/kept*100:.2f}%")
print(f"Multimodal OD bins: {len(mm)} / {s['unique_od_bins']} = {len(mm)/s['unique_od_bins']*100:.3f}%")

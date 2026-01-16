#!/usr/bin/env python3
import json

with open("_sync/wsa/icml2026_routegen/A_mm_od_mioh_v1/report.json") as f:
    d = json.load(f)

cfg = d["scan_config"]
s = d["summary"]
print("=== Config ===")
print(f"od_bin_deg: {cfg['od_bin_deg']} (~{cfg['od_bin_deg']*111:.1f}km)")

print("\n=== Summary ===")
for k, v in s.items():
    print(f"{k}: {v}")

mm = d["multimodal_od_bins"]
print(f"\n=== Multimodal OD Bins (N={len(mm)}) ===")

n_routes = [x["n_routes"] for x in mm]
print(f"n_routes: min={min(n_routes)}, max={max(n_routes)}, median={sorted(n_routes)[len(n_routes)//2]}, mean={sum(n_routes)/len(n_routes):.1f}")
print(f"Total routes in multimodal ODs: {sum(n_routes)}")

jd = [x["top2_jaccard_dist"] for x in mm]
print(f"jaccard_dist: min={min(jd):.2f}, max={max(jd):.2f}, median={sorted(jd)[len(jd)//2]:.2f}")

# n_routes分布
print("\n=== n_routes Distribution ===")
for thr in [5, 10, 15, 20, 30, 50]:
    cnt = sum(1 for n in n_routes if n >= thr)
    routes_sum = sum(n for n in n_routes if n >= thr)
    print(f"  n>={thr:2d}: {cnt:3d} ODs, {routes_sum:4d} routes")

print("\n=== Top 15 by n_routes ===")
od_deg = cfg['od_bin_deg']
for i, x in enumerate(mm[:15]):
    od = x["od_bin"]
    lat_o, lon_o = od[1]*od_deg, od[0]*od_deg
    lat_d, lon_d = od[3]*od_deg, od[2]*od_deg
    print(f"{i+1:2d}. n={x['n_routes']:3d} clusters={str(x['cluster_sizes']):20s} J={x['top2_jaccard_dist']:.2f} O=({lat_o:.2f},{lon_o:.2f}) D=({lat_d:.2f},{lon_d:.2f})")

print(f"\n=== Rate Analysis ===")
kept = s["files_kept_after_filter"]
mm_routes = sum(n_routes)
print(f"Kept trajectories: {kept}")
print(f"Multimodal routes: {mm_routes}")
print(f"Multimodal rate: {mm_routes/kept*100:.2f}%")
print(f"Multimodal OD bins: {len(mm)} / {s['unique_od_bins']} = {len(mm)/s['unique_od_bins']*100:.2f}%")

#!/usr/bin/env python3
import json

# v2 report
with open("_sync/wsa/icml2026_routegen/A_mm_od_mioh_v2_bin02_sep50/report.json") as f:
    d = json.load(f)

cfg = d["scan_config"]
s = d["summary"]
print("=== Config ===")
print(f"od_bin_deg: {cfg['od_bin_deg']} (~{cfg['od_bin_deg']*111:.1f}km)")
print(f"signature_type: {cfg.get('signature_type', 'N/A')}")
print(f"distance_metric: {cfg.get('distance_metric', 'N/A')}")

print("\n=== Summary ===")
for k, v in s.items():
    print(f"{k}: {v}")

mm = d["multimodal_od_bins"]
print(f"\n=== Multimodal OD Bins (N={len(mm)}) ===")

n_routes = [x["n_routes"] for x in mm]
print(f"n_routes: min={min(n_routes)}, max={max(n_routes)}, median={sorted(n_routes)[len(n_routes)//2]}, mean={sum(n_routes)/len(n_routes):.1f}")
print(f"Total routes in multimodal ODs: {sum(n_routes)}")

# LCS distance分布
lcs_dists = [x["top2_lcs_dist"] for x in mm]
print(f"\n=== LCS Distance Distribution ===")
print(f"min={min(lcs_dists):.2f}, max={max(lcs_dists):.2f}, median={sorted(lcs_dists)[len(lcs_dists)//2]:.2f}")

for thr in [0.5, 0.7, 0.9, 1.0]:
    cnt = sum(1 for d in lcs_dists if d >= thr)
    print(f"  LCS dist >= {thr}: {cnt} ODs ({cnt/len(mm)*100:.1f}%)")

# way_seq_lens分布
print(f"\n=== Way Sequence Length ===")
all_lens = []
for x in mm:
    all_lens.extend(x.get("way_seq_lens", []))
if all_lens:
    print(f"way_seq_len: min={min(all_lens)}, max={max(all_lens)}, median={sorted(all_lens)[len(all_lens)//2]}")

# cluster数量分布
print(f"\n=== Cluster Count ===")
n_clusters = [x["n_clusters"] for x in mm]
print(f"n_clusters: min={min(n_clusters)}, max={max(n_clusters)}, median={sorted(n_clusters)[len(n_clusters)//2]}")

print(f"\n=== Top 15 by n_routes ===")
for i, x in enumerate(mm[:15]):
    print(f"{i+1:2d}. n={x['n_routes']:3d} clusters={x['n_clusters']:2d} lcs_dist={x['top2_lcs_dist']:.2f} way_lens={x.get('way_seq_lens', [])[:2]}")

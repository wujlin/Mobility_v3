"""Quick audit: E5seg waypoint-to-GT distance vs E3seg baseline."""
import numpy as np
from scipy.spatial.distance import cdist

# 加载 E5seg 采样结果
e5_samples = np.load('_sync/wsA/icml2026_routegen/E5seg_case00_sample_K20_res0p1_seed0/samples.npz')
print('E5seg keys:', list(e5_samples.keys()))

# 加载 GT case
gt_case = np.load('_sync/wsa/icml2026_routegen/E0s_gt_baseline_detroit_segF256_seed0_od128_min10_sep2_uni2_png/case_00/gt_case.npz')
print('GT keys:', list(gt_case.keys()))

# waypoint 形状
wp_abs = e5_samples['wp_abs_k']
print('wp_abs_k shape:', wp_abs.shape)  # (N, K_samples, K_wp, 2)

# GT targets
gt_targets = gt_case['targets']
print('GT targets shape:', gt_targets.shape)

# 计算 waypoint 到 GT polyline 的距离
N, K_samples, K_wp, _ = wp_abs.shape
dists = []

for i in range(N):
    gt_poly = gt_targets[i]  # (F, 2)
    for k in range(K_samples):
        for w in range(K_wp):
            wp = wp_abs[i, k, w]  # (2,)
            d = cdist(wp.reshape(1, 2), gt_poly).min()
            dists.append(d)

dists = np.array(dists)
print(f'\n=== E5seg (充分训练, 17500 updates) ===')
print(f'Waypoint-to-GT distance:')
print(f'  mean: {dists.mean():.2f}')
print(f'  p50:  {np.percentile(dists, 50):.2f}')
print(f'  p90:  {np.percentile(dists, 90):.2f}')

# 对比基线 (从之前的 audit 结果)
print(f'\n=== E3seg (欠训练, ~200 updates) ===')
print(f'Waypoint-to-GT distance (from previous audit):')
print(f'  mean: 59.37')
print(f'  p50:  49.51')
print(f'  p90:  125.19')

print(f'\n=== E3seg + tier-road (欠训练) ===')
print(f'Waypoint-to-GT distance (from previous audit):')
print(f'  mean: 35.00')
print(f'  p50:  24.70')
print(f'  p90:  83.03')

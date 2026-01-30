import json
import numpy as np

with open('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD_PASTCTX_strict_sem5_rustbelt_seed0/W6_train_ae_pastctx_k8/oracle_step_diagnose/report.json') as f:
    d = json.load(f)

# 分析首次偏离时的 hop 变化
hop_pred_minus_gt = []
gt_hop_at_div = []
for r in d['per_route']:
    ft = r.get('first_div_transition')
    if ft and ft.get('hop_cur') is not None:
        hop_cur = ft['hop_cur']
        hop_pred = ft.get('hop_pred_next', -1)
        if hop_pred >= 0 and hop_cur >= 0:
            hop_pred_minus_gt.append(hop_pred - (hop_cur - 1))
            gt_hop_at_div.append(hop_cur)

arr = np.array(hop_pred_minus_gt)
print(f"首次偏离时 pred_hop 相对 GT 的增量:")
print(f"  mean: {np.mean(arr):.2f}")
print(f"  median: {np.median(arr):.2f}")
print(f"  p90: {np.percentile(arr, 90):.2f}")
print(f"  范围: [{arr.min()}, {arr.max()}]")
print(f"  pred_hop > gt_hop 的比例: {np.mean(arr > 0):.2%}")

gt_hop_arr = np.array(gt_hop_at_div)
print(f"\n偏离发生时距终点 hop 距离:")
print(f"  median: {np.median(gt_hop_arr):.0f}")
print(f"  p90: {np.percentile(gt_hop_arr, 90):.0f}")

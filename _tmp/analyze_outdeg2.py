import json
import numpy as np
from collections import Counter

with open('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD_PASTCTX_strict_sem5_rustbelt_seed0/W6_train_ae_pastctx_k8/oracle_step_diagnose/report.json') as f:
    d = json.load(f)

# 统计 outdeg=2 时的详细情况
outdeg2_cases = []
for r in d['per_route']:
    ft = r.get('first_div_transition')
    if ft and ft.get('succ_full_n') == 2:
        outdeg2_cases.append({
            'hop_cur': ft['hop_cur'],
            'hop_pred': ft['hop_pred_next'],
            'gt_gap': ft['gt_gap'],
            'close_call': ft['close_call'],
            'gt_rank': ft['gt_rank'],
        })

print(f"=== 二选一场景分析 (outdeg=2, n={len(outdeg2_cases)}) ===")

# 在 outdeg=2 时，GT 理论上应该让 hop-1
hop_deltas = []
for c in outdeg2_cases:
    if c['hop_pred'] >= 0 and c['hop_cur'] >= 0:
        # GT 选择让 hop_cur -> hop_cur-1
        # pred 选择让 hop_cur -> hop_pred
        # 如果 hop_pred > hop_cur-1，说明选了"走弯路"的方向
        delta = c['hop_pred'] - (c['hop_cur'] - 1)
        hop_deltas.append(delta)

arr = np.array(hop_deltas)
print(f"pred 相对 GT 的 hop 增量:")
print(f"  =0 (都减1，方向一致): {np.mean(arr == 0):.1%}")
print(f"  =1 (不变 vs 减1): {np.mean(arr == 1):.1%}")
print(f"  >1 (明显走弯路): {np.mean(arr > 1):.1%}")
print(f"  median: {np.median(arr):.0f}, max: {arr.max()}")

# gt_gap 和 hop_delta 的关系
print(f"\ngt_gap (logit差距) vs hop_delta:")
for c in outdeg2_cases[:10]:
    delta = c['hop_pred'] - (c['hop_cur'] - 1) if c['hop_pred'] >= 0 else None
    print(f"  gt_rank={c['gt_rank']}, gt_gap={c['gt_gap']:.2f}, hop_delta={delta}, close={c['close_call']}")

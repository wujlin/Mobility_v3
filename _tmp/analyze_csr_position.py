"""
深层分析：GT rank=2 的真正原因

已排除的假设：
1. CSR 顺序偏差 - CSR 按 way_id 升序，但 pred_smaller=43%，接近随机
2. dest_dist shortcut - pred_closer_to_dest=44%，接近随机
3. 方向信息缺失 - Direction Hint 提升了成功率，但没有改变 rank=2

新的观察：
- 在 outdeg=2 时，GT rank=2 的比例是 **100%**
- 这意味着模型学到了某种系统性偏好，而 GT 恰好**总是不符合**这个偏好

让我检查一个关键问题：训练数据中，GT 在位置 0 还是位置 1？
"""

import json
import numpy as np

# 加载 B 组数据
with open('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD_AB_dirhint_B_dirq_strict_sem5_seed0_e100/W6_train_ae/oracle_step_n200.json') as f:
    B = json.load(f)

# outdeg=2 且 GT rank=2 意味着：
# - 评估时 CSR 顺序是 [pred, gt]
# - 模型选了位置 0 (pred)，GT 在位置 1
# 
# 但训练时，GT 在候选中的位置是由 _succ_cands 的返回顺序决定的：
# - _succ_cands 返回 CSR 顺序的邻居
# - _ensure_target 如果 GT 已在其中，则不改变顺序
#
# 关键问题：在评估的这些失败案例中，GT 为什么总是在 CSR 位置 1？

# 检查 way_id 关系
cases = []
for r in B['per_route']:
    ft = r.get('first_div_transition')
    if ft and ft.get('succ_full_n') == 2 and ft.get('gt_rank') == 2:
        cases.append({
            'cur_way': ft['cur_way'],
            'gt_next': ft['gt_next'],
            'pred_next': ft['pred_next'],
        })

# CSR 按 way_id 升序，所以：
# - 如果 pred < gt，说明 pred 在 CSR 位置 0
# - 如果 pred > gt，说明 pred 在 CSR 位置 1（但模型仍选了它）

pred_is_csr_pos0 = sum(1 for c in cases if c['pred_next'] < c['gt_next'])
pred_is_csr_pos1 = sum(1 for c in cases if c['pred_next'] > c['gt_next'])

print(f"总案例数: {len(cases)}")
print(f"pred 在 CSR 位置 0 (pred_id < gt_id): {pred_is_csr_pos0} ({pred_is_csr_pos0/len(cases):.1%})")
print(f"pred 在 CSR 位置 1 (pred_id > gt_id): {pred_is_csr_pos1} ({pred_is_csr_pos1/len(cases):.1%})")

print(f"""
=== 关键洞察 ===

如果 pred_is_csr_pos0 ≈ 100%：
  → 模型总是选 CSR 位置 0（位置偏差）
  → 而 GT 总是在 CSR 位置 1
  → 这说明训练数据中 GT 总是在位置 0，模型学到了"选位置0"

如果 pred_is_csr_pos1 ≈ 100%：
  → 模型总是选 CSR 位置 1
  → 这更奇怪，需要进一步分析

如果接近 50%：
  → 模型不是按位置选择
  → 而是学到了某种语义偏好（如道路类型）
""")

# 更深入：检查这些案例中 pred 和 gt 的语义特征差异
# 这需要加载 way_features，但我们可以从诊断数据推断一些信息

# 检查 hop 行为
hop_pred_better = 0
hop_gt_better = 0
for c in cases:
    # 这里没有 hop 信息，需要从完整数据获取
    pass

# 检查距离
dist_pred_closer = 0
dist_gt_closer = 0
for r in B['per_route']:
    ft = r.get('first_div_transition')
    if ft and ft.get('succ_full_n') == 2 and ft.get('gt_rank') == 2:
        dp = ft.get('dist_pred_to_dest')
        dg = ft.get('dist_gt_to_dest')
        if dp is not None and dg is not None:
            if dp < dg:
                dist_pred_closer += 1
            else:
                dist_gt_closer += 1

print(f"\n=== 欧氏距离分析 ===")
print(f"pred 离终点更近: {dist_pred_closer} ({dist_pred_closer/(dist_pred_closer+dist_gt_closer):.1%})")
print(f"gt 离终点更近: {dist_gt_closer} ({dist_gt_closer/(dist_pred_closer+dist_gt_closer):.1%})")

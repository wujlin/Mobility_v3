"""
位置偏差的代码逻辑分析

关键代码路径（way_sequence_dataset.py）：

1. _succ_cands(way):
   - 从 CSR 图中取出后继：way_adj_idx[ptr[way]:ptr[way+1]]
   - 顺序是**确定性的**（由图构建时决定）

2. _ensure_target(cands, target):
   - 如果 target 已在 cands 中 → 返回原序（不打乱）
   - 如果 target 不在 → 插入末位

3. 训练时：
   - cands = _succ_cands(prev)[:max_candidates]
   - cands = _ensure_target(cands, tgt)
   - pos = np.where(row == tgt)[0][0]  # GT 的位置

关键问题：如果 CSR 图的存储顺序使得 GT 经常在位置 0，
模型会学到 "位置 0 偏好"。

让我们从诊断数据反推：
- 评估时 GT rank=2 恒定（outdeg=2）
- 这意味着 GT 在 CSR 中是"第二个"，但模型选了"第一个"
- 说明训练时 GT 经常在"第一个"位置
"""

import json
import numpy as np

# 从诊断数据中分析
with open('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD_AB_dirhint_B_dirq_strict_sem5_seed0_e100/W6_train_ae/oracle_step_n200.json') as f:
    B = json.load(f)

# 收集 outdeg=2 且 GT rank=2 的案例
cases = []
for r in B['per_route']:
    ft = r.get('first_div_transition')
    if ft and ft.get('succ_full_n') == 2 and ft.get('gt_rank') == 2:
        cases.append({
            'cur_way': ft['cur_way'],
            'gt_next': ft['gt_next'],
            'pred_next': ft['pred_next'],
        })

print(f"outdeg=2 且 GT rank=2 的案例: {len(cases)}")

# 检查 pred_next vs gt_next 的 way_id 大小关系
pred_smaller = sum(1 for c in cases if c['pred_next'] < c['gt_next'])
print(f"pred_way_id < gt_way_id: {pred_smaller}/{len(cases)} = {pred_smaller/len(cases):.1%}")

# 假设：CSR 按 way_id 升序存储邻居
# 如果 pred_smaller ≈ 50%，说明不是 way_id 顺序
# 如果 pred_smaller >> 50% 或 << 50%，说明有系统性偏差

print(f"""
分析：
- 如果 CSR 按 way_id 升序存储，且模型偏好位置 0，那么 pred 应该是较小的 way_id
- 但数据显示 pred_smaller = 43%，接近随机
- 这说明 CSR 顺序不是纯 way_id 升序

真正的问题可能在于：
1. 图构建时的遍历顺序（如 networkx 的 neighbors 顺序）
2. 或者模型确实学到了某种候选内的位置偏好
""")

# 进一步验证：检查 gt_gap 和 位置的关系
# gt_gap 大说明模型很自信地选错了
gaps = []
for r in B['per_route']:
    ft = r.get('first_div_transition')
    if ft and ft.get('succ_full_n') == 2:
        gaps.append(ft['gt_gap'])

print(f"\n=== outdeg=2 时的 gt_gap 分布 ===")
print(f"  mean: {np.mean(gaps):.3f}")
print(f"  median: {np.median(gaps):.3f}")
print(f"  p90: {np.percentile(gaps, 90):.3f}")

# 最关键的问题：模型的 scorer 是否有位置偏好？
print(f"""
=== 核心假设 ===

假设：训练数据中 target 的位置分布不均匀

验证方法：
1. 在工作站上统计训练数据的 target_idx 分布
2. 如果 target_idx=0 的比例 >> 50%，证实位置偏差

修复方法：
1. 训练时随机打乱候选顺序
2. 或者使用位置无关的打分（如 set-based scoring）
""")

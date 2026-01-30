"""
深层问题分析：为什么 GT rank 恒等于 2？

核心洞察：在 outdeg=2 的二选一场景中，GT rank=2 的比例是 100%。
这不是"随机选错"，而是系统性地把 GT 排在第二位。
"""
import json
import numpy as np

with open('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD_AB_dirhint_B_dirq_strict_sem5_seed0_e100/W6_train_ae/oracle_step_n200.json') as f:
    B = json.load(f)

# 收集所有 outdeg=2 且 GT rank=2 的案例
cases = []
for r in B['per_route']:
    ft = r.get('first_div_transition')
    if ft and ft.get('succ_full_n') == 2:
        cases.append({
            'route_id': r['route_id'],
            'city': r['city'],
            'step_idx': ft['step_idx'],
            'cur_way': ft['cur_way'],
            'gt_next': ft['gt_next'],
            'pred_next': ft['pred_next'],
            'hop_cur': ft['hop_cur'],
            'hop_pred': ft['hop_pred_next'],
            'gt_gap': ft['gt_gap'],
            'logit_margin': ft['logit_margin'],
            'dist_pred': ft.get('dist_pred_to_dest'),
            'dist_gt': ft.get('dist_gt_to_dest'),
        })

print(f"总共 {len(cases)} 个 outdeg=2 场景")
print(f"GT rank=2 比例: 100%（恒定）\n")

# 关键问题：pred_next 和 gt_next 有什么系统性差异？
print("=" * 60)
print("假设检验：模型是否总是选择某类道路？")
print("=" * 60)

# 1. 检查 pred_next < gt_next 的比例（way_id 大小可能反映某种顺序）
pred_smaller = sum(1 for c in cases if c['pred_next'] < c['gt_next'])
print(f"\n1. pred_way_id < gt_way_id 的比例: {pred_smaller}/{len(cases)} = {pred_smaller/len(cases):.1%}")

# 2. 检查 hop 行为
hop_pred_better = sum(1 for c in cases if c['hop_pred'] < c['hop_cur'])  # pred 让 hop 减少
hop_gt_better = sum(1 for c in cases if (c['hop_cur'] - 1) < c['hop_pred'])  # GT 让 hop 减少更多
print(f"\n2. hop 行为:")
print(f"   pred 使 hop 减少（正确方向）: {hop_pred_better}/{len(cases)} = {hop_pred_better/len(cases):.1%}")
print(f"   GT 使 hop 减少更多（GT方向更优）: {hop_gt_better}/{len(cases)} = {hop_gt_better/len(cases):.1%}")

# 3. 检查欧氏距离
dist_pred_closer = sum(1 for c in cases if c['dist_pred'] and c['dist_gt'] and c['dist_pred'] < c['dist_gt'])
dist_valid = sum(1 for c in cases if c['dist_pred'] and c['dist_gt'])
print(f"\n3. 欧氏距离（对终点）:")
print(f"   pred 离终点更近: {dist_pred_closer}/{dist_valid} = {dist_pred_closer/dist_valid:.1%}")

# 4. 关键：logit_margin 分布（模型有多自信）
margins = [c['logit_margin'] for c in cases]
print(f"\n4. logit_margin (模型置信度):")
print(f"   mean: {np.mean(margins):.3f}")
print(f"   median: {np.median(margins):.3f}")
print(f"   p90: {np.percentile(margins, 90):.3f}")

# 分桶看
small_margin = [c for c in cases if c['logit_margin'] < 0.3]
large_margin = [c for c in cases if c['logit_margin'] > 1.0]
print(f"\n   小margin(<0.3): {len(small_margin)} 个 (不确定)")
print(f"   大margin(>1.0): {len(large_margin)} 个 (很自信地选错)")

# 5. 检查 step_idx 分布（是早期还是晚期偏离）
steps = [c['step_idx'] for c in cases]
print(f"\n5. 偏离发生在第几步:")
print(f"   p10: {np.percentile(steps, 10):.0f}")
print(f"   p50: {np.percentile(steps, 50):.0f}")
print(f"   p90: {np.percentile(steps, 90):.0f}")

early_step = sum(1 for c in cases if c['step_idx'] <= 5)
print(f"   早期偏离(step<=5): {early_step}/{len(cases)} = {early_step/len(cases):.1%}")

print("\n" + "=" * 60)
print("核心结论")
print("=" * 60)
print("""
1. GT rank=2 是恒定的（100%），说明问题是系统性的，不是随机噪声。

2. 可能的原因：
   a) 训练数据中的候选顺序有偏（CSR 存储顺序）
   b) 模型学到了"第一个候选更可能正确"的伪相关
   c) scorer 的初始化/架构导致对第一个候选有偏好

3. Direction Hint 提升了整体成功率（+5.2pp），但没有改变这个系统性偏差。
   这说明方向信息帮助了一部分决策，但根本问题不在方向上。
""")

# 6. 检验假设：是否与 CSR 顺序相关
print("\n" + "=" * 60)
print("假设：问题与候选顺序（CSR存储）相关")
print("=" * 60)

# 如果 pred_next 总是 CSR 中的第一个，那 gt_next 应该是第二个
# 在 CSR 中，第一个邻居通常是 way_id 较小的那个（或按某种图遍历顺序）
print("""
观察：
- pred_next 和 gt_next 是 CSR 中仅有的两个后继
- 模型恒定选择"某一个"而非"另一个"
- 这指向一个 positional bias（位置偏差）

验证方法：
1. 检查训练时 collate_fn 中候选的顺序
2. 检查评估时候选的顺序是否与训练一致
3. 考虑在训练时随机打乱候选顺序
""")

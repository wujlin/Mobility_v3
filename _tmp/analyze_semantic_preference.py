"""
终极分析：模型学到了什么样的语义偏好？

已确认的事实：
1. GT rank=2 恒定（100%）
2. 不是 CSR 位置偏差（pred 在位置 0/1 各占 43%/57%）
3. 不是欧氏距离偏好（pred closer 45.6%，接近随机）
4. Direction Hint 有帮助但没有改变核心症状

新假设：模型学到了"继续直行"vs"转弯"的偏好

验证：如果模型偏好"直行"，那么在分叉点，它应该选择与当前方向更一致的那条路
"""

import json
import numpy as np

with open('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD_AB_dirhint_B_dirq_strict_sem5_seed0_e100/W6_train_ae/oracle_step_n200.json') as f:
    B = json.load(f)

# 从 focus_traces 中获取更详细的信息
print("=== 从 focus_traces 分析 ===")
print(f"focus_traces 数量: {len(B['focus_traces'])}")

# 分析 hop 变化模式
hop_deltas = []
for r in B['per_route']:
    ft = r.get('first_div_transition')
    if ft and ft.get('succ_full_n') == 2 and ft.get('hop_cur') is not None and ft.get('hop_pred_next') is not None:
        hop_cur = ft['hop_cur']
        hop_pred = ft['hop_pred_next']
        # GT 应该让 hop 减 1
        delta = hop_pred - (hop_cur - 1)
        hop_deltas.append({
            'delta': delta,
            'hop_cur': hop_cur,
            'hop_pred': hop_pred,
            'gt_gap': ft['gt_gap'],
        })

print(f"\n=== hop 行为深度分析 (outdeg=2, n={len(hop_deltas)}) ===")

# delta 分布
deltas = [h['delta'] for h in hop_deltas]
print(f"hop_delta 分布:")
print(f"  delta=0 (pred也减1): {sum(1 for d in deltas if d == 0)/len(deltas):.1%}")
print(f"  delta=1 (pred不变): {sum(1 for d in deltas if d == 1)/len(deltas):.1%}")  
print(f"  delta>1 (pred增加，走弯路): {sum(1 for d in deltas if d > 1)/len(deltas):.1%}")
print(f"  delta<0 (pred比GT更好?): {sum(1 for d in deltas if d < 0)/len(deltas):.1%}")

# 关键：当 delta=0 时，GT 也在减少 hop，为什么还是选错？
delta_0_cases = [h for h in hop_deltas if h['delta'] == 0]
print(f"\n=== delta=0 的案例 (n={len(delta_0_cases)}) ===")
print("  这些案例中，pred 和 GT 都让 hop 减 1，但模型仍选 pred")
print("  这说明模型不是基于 hop/方向，而是基于某种其他特征")

# 检查 delta=0 案例的 gt_gap
gaps_d0 = [h['gt_gap'] for h in delta_0_cases]
print(f"  gt_gap 分布: mean={np.mean(gaps_d0):.3f}, median={np.median(gaps_d0):.3f}")

# 最关键的问题：模型到底在看什么？
print(f"""
=== 核心推断 ===

观察：
1. 模型不是按 CSR 位置选择（排除位置偏差）
2. 模型不是按欧氏距离选择（pred closer 45.6%）
3. 即使 pred 和 GT 都让 hop 减 1（方向正确），模型仍选 pred

结论：
模型学到了某种**道路级别的偏好**，而不是位置/方向/距离。

可能的偏好类型：
1. road_tier 偏好（主干道 vs 次干道）
2. highway_type 偏好（如偏好 residential 而非 service）
3. 道路长度偏好（长路 vs 短路）
4. 训练数据中的道路频率偏好（常见道路 vs 罕见道路）

验证方法：
需要加载 way_features，对比 pred 和 gt 的语义特征差异。
""")

# 额外分析：检查成功案例 vs 失败案例的特征差异
success_cases = [r for r in B['per_route'] if r['success']]
fail_cases = [r for r in B['per_route'] if not r['success']]

print(f"\n=== 成功 vs 失败案例对比 ===")
print(f"成功: {len(success_cases)} 条 ({len(success_cases)/len(B['per_route']):.1%})")
print(f"失败: {len(fail_cases)} 条 ({len(fail_cases)/len(B['per_route']):.1%})")

# 成功案例中，exact match 的比例
exact_in_success = sum(1 for r in success_cases if r['seq_exact'])
print(f"成功中 exact match: {exact_in_success}/{len(success_cases)} ({exact_in_success/len(success_cases):.1%})")

# 失败案例的 diverge_idx 分布
div_idx = [r['diverge_idx'] for r in fail_cases if r['diverge_idx'] is not None]
print(f"\n失败案例的首次偏离位置:")
print(f"  p10: {np.percentile(div_idx, 10):.0f}")
print(f"  p50: {np.percentile(div_idx, 50):.0f}")
print(f"  p90: {np.percentile(div_idx, 90):.0f}")

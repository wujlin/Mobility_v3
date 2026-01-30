"""
核心问题分析：为什么模型在二选一时总是选错？

假设检验：模型是否学到了"拓扑先验"而非"路线意图"
"""
import json
import numpy as np

# 加载数据
with open('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD_PASTCTX_strict_sem5_rustbelt_seed0/W6_train_ae_pastctx_k8/oracle_step_diagnose/report.json') as f:
    d = json.load(f)

# 收集所有首次偏离的案例
div_cases = []
for r in d['per_route']:
    ft = r.get('first_div_transition')
    if ft:
        div_cases.append({
            'city': r['city'],
            'step_idx': ft['step_idx'],
            'outdeg': ft['succ_full_n'],
            'hop_cur': ft['hop_cur'],
            'hop_pred': ft['hop_pred_next'],
            'gt_rank': ft['gt_rank'],
            'gt_gap': ft['gt_gap'],
            'logit_margin': ft['logit_margin'],
            'close_call': ft['close_call'],
        })

print(f"=== 首次偏离分析 (n={len(div_cases)}) ===\n")

# 关键问题1: gt_rank 分布
gt_ranks = [c['gt_rank'] for c in div_cases]
print(f"GT 排名分布:")
for rank in sorted(set(gt_ranks)):
    frac = sum(1 for r in gt_ranks if r == rank) / len(gt_ranks)
    print(f"  rank={rank}: {frac:.1%}")

# 关键问题2: 高置信错误 (gt_gap > 1 且 hop_delta > 5)
high_conf_wrong = []
for c in div_cases:
    hop_delta = c['hop_pred'] - (c['hop_cur'] - 1) if c['hop_pred'] >= 0 and c['hop_cur'] >= 0 else 0
    if c['gt_gap'] > 1.0 and hop_delta > 5:
        high_conf_wrong.append(c)

print(f"\n高置信错误 (gt_gap>1 且 hop_delta>5): {len(high_conf_wrong)}/{len(div_cases)} = {len(high_conf_wrong)/len(div_cases):.1%}")

# 关键问题3: step 位置 vs 偏离
step_idx_list = [c['step_idx'] for c in div_cases]
print(f"\n偏离发生的 step 位置:")
print(f"  p10: {np.percentile(step_idx_list, 10):.0f}")
print(f"  p50: {np.percentile(step_idx_list, 50):.0f}")
print(f"  p90: {np.percentile(step_idx_list, 90):.0f}")

# 关键问题4: 早期偏离 vs 晚期偏离
early_div = [c for c in div_cases if c['step_idx'] <= 5]
late_div = [c for c in div_cases if c['step_idx'] > 20]
print(f"\n早期偏离 (step<=5): {len(early_div)} 案例")
print(f"  close_call 比例: {np.mean([c['close_call'] for c in early_div]) if early_div else 0:.1%}")
print(f"晚期偏离 (step>20): {len(late_div)} 案例")
print(f"  close_call 比例: {np.mean([c['close_call'] for c in late_div]) if late_div else 0:.1%}")

# 关键问题5: 分城市
print(f"\n分城市统计:")
for city in [0, 1]:
    city_cases = [c for c in div_cases if c['city'] == city]
    city_name = "Detroit" if city == 0 else "Columbus"
    print(f"  {city_name}: n={len(city_cases)}, 平均gt_gap={np.mean([c['gt_gap'] for c in city_cases]):.2f}")

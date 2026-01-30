"""
Direction Hint A/B 实验深度分析
"""
import json
import numpy as np

# 加载两组数据
with open('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD_AB_dirhint_A_nodir_strict_sem5_seed0_e100/W6_train_ae/oracle_step_n200.json') as f:
    A = json.load(f)

with open('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD_AB_dirhint_B_dirq_strict_sem5_seed0_e100/W6_train_ae/oracle_step_n200.json') as f:
    B = json.load(f)

print("=" * 60)
print("Direction Hint A/B 实验对比分析")
print("=" * 60)

# 1. 整体成功率对比
print("\n【1. 整体指标】")
print(f"  成功率: A={A['summary']['success_rate']:.3f} → B={B['summary']['success_rate']:.3f} (Δ={B['summary']['success_rate']-A['summary']['success_rate']:+.3f})")
print(f"  精确匹配率: A={A['summary']['success_exact_rate']:.3f} → B={B['summary']['success_exact_rate']:.3f}")

# 2. GT rank 分布对比
print("\n【2. 首次偏离时 GT 排名】")
for key in ['p00', 'p50', 'p90', 'p95', 'p100']:
    a_val = A['q2_logits']['first_div_gt_rank_quantiles'][key]
    b_val = B['q2_logits']['first_div_gt_rank_quantiles'][key]
    print(f"  {key}: A={a_val} → B={b_val}")

# 3. close_call 对比
print("\n【3. Close-call 比例】")
print(f"  A: {A['q2_logits']['first_div_close_call_frac']:.1%}")
print(f"  B: {B['q2_logits']['first_div_close_call_frac']:.1%}")

# 4. 分析 per_route 级别的变化
a_routes = {r['route_id']: r for r in A['per_route']}
b_routes = {r['route_id']: r for r in B['per_route']}

common_ids = set(a_routes.keys()) & set(b_routes.keys())
print(f"\n【4. 路线级对比 (n={len(common_ids)})】")

# A失败→B成功
a_fail_b_succ = []
# A成功→B失败
a_succ_b_fail = []
# 两者都失败但B偏离更晚
both_fail_b_better = []

for rid in common_ids:
    ra, rb = a_routes[rid], b_routes[rid]
    if not ra['success'] and rb['success']:
        a_fail_b_succ.append(rid)
    elif ra['success'] and not rb['success']:
        a_succ_b_fail.append(rid)
    elif not ra['success'] and not rb['success']:
        div_a = ra.get('diverge_idx') or 0
        div_b = rb.get('diverge_idx') or 0
        if div_b > div_a:
            both_fail_b_better.append((rid, div_a, div_b))

print(f"  A失败→B成功: {len(a_fail_b_succ)} 条")
print(f"  A成功→B失败: {len(a_succ_b_fail)} 条")
print(f"  两者都失败但B偏离更晚: {len(both_fail_b_better)} 条")

# 5. 分析首次偏离的 hop 变化
print("\n【5. 首次偏离时 hop 行为对比】")

def extract_hop_stats(data):
    hop_deltas = []
    for r in data['per_route']:
        ft = r.get('first_div_transition')
        if ft and ft.get('hop_cur') is not None and ft.get('hop_pred_next') is not None:
            hop_cur = ft['hop_cur']
            hop_pred = ft['hop_pred_next']
            # GT 应该让 hop 减 1
            delta = hop_pred - (hop_cur - 1)
            hop_deltas.append(delta)
    return np.array(hop_deltas)

hop_a = extract_hop_stats(A)
hop_b = extract_hop_stats(B)

print(f"  A: 平均hop_delta={np.mean(hop_a):.2f}, 走弯路比例(delta>1)={np.mean(hop_a>1):.1%}")
print(f"  B: 平均hop_delta={np.mean(hop_b):.2f}, 走弯路比例(delta>1)={np.mean(hop_b>1):.1%}")

# 6. 分析 outdeg=2 的二选一场景
print("\n【6. 二选一场景 (outdeg=2) 详细分析】")

def analyze_outdeg2(data, name):
    cases = []
    for r in data['per_route']:
        ft = r.get('first_div_transition')
        if ft and ft.get('succ_full_n') == 2:
            cases.append({
                'gt_rank': ft['gt_rank'],
                'close_call': ft['close_call'],
                'gt_gap': ft['gt_gap'],
            })
    
    if cases:
        gt_rank_1 = sum(1 for c in cases if c['gt_rank'] == 1)
        gt_rank_2 = sum(1 for c in cases if c['gt_rank'] == 2)
        close = sum(1 for c in cases if c['close_call'])
        avg_gap = np.mean([c['gt_gap'] for c in cases])
        print(f"  {name}: n={len(cases)}, GT_rank=1:{gt_rank_1}({gt_rank_1/len(cases):.1%}), rank=2:{gt_rank_2}({gt_rank_2/len(cases):.1%}), close_call:{close/len(cases):.1%}, avg_gap:{avg_gap:.2f}")

analyze_outdeg2(A, "A(无dir)")
analyze_outdeg2(B, "B(有dir)")

# 7. 关键问题：为什么 GT rank 始终是 2？
print("\n【7. 深层分析：GT rank=2 的案例特征】")
print("  (检查是否存在某种系统性偏好)")

# 从 B 组中抽取几个 GT rank=2 的案例
rank2_cases = []
for r in B['per_route']:
    ft = r.get('first_div_transition')
    if ft and ft.get('gt_rank') == 2 and ft.get('succ_full_n') == 2:
        rank2_cases.append({
            'route_id': r['route_id'],
            'city': r['city'],
            'step_idx': ft['step_idx'],
            'hop_cur': ft['hop_cur'],
            'hop_pred': ft['hop_pred_next'],
            'gt_gap': ft['gt_gap'],
            'dist_pred': ft.get('dist_pred_to_dest'),
            'dist_gt': ft.get('dist_gt_to_dest'),
        })

print(f"  B组中 outdeg=2 且 GT_rank=2 的案例: {len(rank2_cases)} 个")
print("  抽样5个:")
for c in rank2_cases[:5]:
    hop_delta = c['hop_pred'] - (c['hop_cur'] - 1) if c['hop_pred'] and c['hop_cur'] else None
    dist_diff = (c['dist_pred'] - c['dist_gt']) if c['dist_pred'] and c['dist_gt'] else None
    print(f"    route={c['route_id']}, step={c['step_idx']}, hop_delta={hop_delta}, gt_gap={c['gt_gap']:.2f}, dist_diff={dist_diff:.1f}m" if dist_diff else f"    route={c['route_id']}, step={c['step_idx']}, hop_delta={hop_delta}, gt_gap={c['gt_gap']:.2f}")

"""分析 Way-CASD Oracle Decode 失败诊断报告"""
import json
from collections import Counter
import statistics

def percentile(data, p):
    """计算百分位数"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (n - 1) * p / 100
    lower = int(idx)
    upper = lower + 1
    if upper >= n:
        return sorted_data[-1]
    weight = idx - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

# 读取报告
report_path = r"e:\newdesktop\HKUST\GeoExplicit_SFM\v3\_sync\wsa\icml2026_routegen\WAYCASD_STRICT_DIAG_oracle_fail_v1\gt_greedy_all\report.json"

with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

print("=" * 80)
print("Way-CASD Oracle Decode 失败诊断报告 - 统计分析")
print("=" * 80)
print(f"\n任务: {report['task']}")
print(f"创建时间: {report['created_at']}")
print(f"评估路线数/城市: {report['cfg']['n_routes']}")
print(f"解码策略: {report['cfg']['decode']}")
print(f"最大解码长度: {report['cfg']['max_decode_len']}")

for city_data in report['per_city']:
    city_id = city_data['city']
    print("\n" + "=" * 80)
    print(f"城市 {city_id} 统计")
    print("=" * 80)
    
    # 基础统计
    print(f"\n【基础指标】")
    print(f"  评估样本数: {city_data['n_eval']}")
    print(f"  成功率 (success_rate): {city_data['success_rate']:.2%}")
    print(f"  撞墙率 (hit_wall_rate): {city_data['hit_wall_rate']:.2%}")
    print(f"  死胡同率 (dead_end_rate): {city_data['dead_end_rate']:.2%}")
    print(f"  成功案例数: {city_data['successes_n']}")
    print(f"  失败案例数: {city_data['failures_n']}")
    
    failures = city_data['failures']
    if not failures:
        continue
    
    # 计算各项统计
    diverge_steps = [f['diverge_step'] for f in failures]
    prefix_match_lens = [f['prefix_match_len'] for f in failures]
    diverge_can_reach = [f['diverge_pred_can_reach_dest'] for f in failures]
    rejoin_gt = [f['rejoin_gt_after_diverge'] for f in failures]
    any_dest_in_full = [f['last_k']['any_dest_in_full'] for f in failures]
    gt_final_in_full = [f['gt_final_hop']['gt_final_in_full'] for f in failures]
    gt_final_in_sel = [f['gt_final_hop']['gt_final_in_sel'] for f in failures]
    
    print(f"\n【发散步数 diverge_step 分布】")
    print(f"  最小值: {min(diverge_steps)}")
    print(f"  p10: {percentile(diverge_steps, 10):.1f}")
    print(f"  p25: {percentile(diverge_steps, 25):.1f}")
    print(f"  p50 (中位数): {percentile(diverge_steps, 50):.1f}")
    print(f"  p75: {percentile(diverge_steps, 75):.1f}")
    print(f"  p90: {percentile(diverge_steps, 90):.1f}")
    print(f"  最大值: {max(diverge_steps)}")
    print(f"  均值: {statistics.mean(diverge_steps):.2f}")
    
    print(f"\n【前缀匹配长度 prefix_match_len 分布】")
    print(f"  最小值: {min(prefix_match_lens)}")
    print(f"  p10: {percentile(prefix_match_lens, 10):.1f}")
    print(f"  p25: {percentile(prefix_match_lens, 25):.1f}")
    print(f"  p50 (中位数): {percentile(prefix_match_lens, 50):.1f}")
    print(f"  p75: {percentile(prefix_match_lens, 75):.1f}")
    print(f"  p90: {percentile(prefix_match_lens, 90):.1f}")
    print(f"  最大值: {max(prefix_match_lens)}")
    
    print(f"\n【关键比例指标】")
    print(f"  diverge_pred_can_reach_dest (发散后仍可达目的地): {sum(diverge_can_reach)/len(diverge_can_reach):.2%}")
    print(f"  rejoin_gt_after_diverge (发散后重新汇入GT轨迹): {sum(rejoin_gt)/len(rejoin_gt):.2%}")
    print(f"  any_dest_in_full (last_k中dest曾出现在full候选): {sum(any_dest_in_full)/len(any_dest_in_full):.2%}")
    print(f"  gt_final_in_full (GT最终跳在full候选中): {sum(gt_final_in_full)/len(gt_final_in_full):.2%}")
    print(f"  gt_final_in_sel (GT最终跳在sel候选中): {sum(gt_final_in_sel)/len(gt_final_in_sel):.2%}")
    
    # 分析hit_wall案例的振荡模式
    hit_wall_cases = [f for f in failures if f['hit_wall']]
    print(f"\n【Hit Wall 案例分析】 (共 {len(hit_wall_cases)} 个)")
    
    oscillation_count = 0
    for case in hit_wall_cases:
        steps = case['last_k']['steps']
        if len(steps) >= 2:
            # 检查振荡模式：连续步骤中cur值交替出现
            curs = [s['cur'] for s in steps]
            if len(set(curs)) == 2:  # 只有两个不同的cur值
                # 检查是否真正振荡
                if len(curs) >= 5:
                    if curs[0] == curs[2] == curs[4]:
                        oscillation_count += 1
                elif len(curs) >= 3:
                    if curs[0] == curs[2]:
                        oscillation_count += 1
    
    print(f"  检测到振荡模式(2节点来回)的案例数: {oscillation_count} ({oscillation_count/len(hit_wall_cases):.2%})")
    
    # 检查更宽松的振荡模式
    stuck_pattern_count = 0
    for case in hit_wall_cases:
        steps = case['last_k']['steps']
        curs = [s['cur'] for s in steps]
        if len(set(curs)) <= 3:  # 最后5步只涉及3个或更少的节点
            stuck_pattern_count += 1
    print(f"  最后5步只涉及<=3个节点: {stuck_pattern_count} ({stuck_pattern_count/len(hit_wall_cases):.2%})")
    
    # 分析dead_end案例
    dead_end_cases = [f for f in failures if f['dead_end']]
    print(f"\n【Dead End 案例分析】 (共 {len(dead_end_cases)} 个)")
    if dead_end_cases:
        de_diverge_steps = [f['diverge_step'] for f in dead_end_cases]
        de_gt_lens = [f['gt_len'] for f in dead_end_cases]
        de_pred_lens = [f['pred_len'] for f in dead_end_cases]
        de_jaccards = [f['jaccard'] for f in dead_end_cases]
        print(f"  发散步数中位数: {statistics.median(de_diverge_steps):.1f}")
        print(f"  GT长度中位数: {statistics.median(de_gt_lens):.1f}")
        print(f"  预测长度中位数: {statistics.median(de_pred_lens):.1f}")
        print(f"  Jaccard中位数: {statistics.median(de_jaccards):.3f}")

print("\n" + "=" * 80)
print("典型失败案例分析")
print("=" * 80)

# 选取典型案例
for city_data in report['per_city']:
    city_id = city_data['city']
    failures = city_data['failures']
    
    print(f"\n--- 城市 {city_id} 典型失败案例 ---")
    
    # 案例1: 早期发散 + hit_wall
    early_diverge_hitwall = [f for f in failures if f['hit_wall'] and f['diverge_step'] <= 3]
    if early_diverge_hitwall:
        case = early_diverge_hitwall[0]
        print(f"\n[类型1: 早期发散 + 撞墙] route_id={case['route_id']}")
        print(f"  GT长度: {case['gt_len']}, 预测长度: {case['pred_len']}")
        print(f"  发散步: {case['diverge_step']}, 前缀匹配: {case['prefix_match_len']}")
        print(f"  Jaccard: {case['jaccard']:.3f}")
        print(f"  发散后可达dest: {case['diverge_pred_can_reach_dest']}")
        steps = case['last_k']['steps']
        curs = [s['cur'] for s in steps]
        print(f"  最后5步cur: {curs}")
    
    # 案例2: 晚期发散 + dead_end
    late_diverge_deadend = [f for f in failures if f['dead_end'] and f['diverge_step'] > 20]
    if late_diverge_deadend:
        case = late_diverge_deadend[0]
        print(f"\n[类型2: 晚期发散 + 死胡同] route_id={case['route_id']}")
        print(f"  GT长度: {case['gt_len']}, 预测长度: {case['pred_len']}")
        print(f"  发散步: {case['diverge_step']}, 前缀匹配: {case['prefix_match_len']}")
        print(f"  Jaccard: {case['jaccard']:.3f}")
        print(f"  发散后可达dest: {case['diverge_pred_can_reach_dest']}")
    
    # 案例3: 高Jaccard但仍失败
    high_jaccard_fail = [f for f in failures if f['jaccard'] > 0.6]
    if high_jaccard_fail:
        case = sorted(high_jaccard_fail, key=lambda x: -x['jaccard'])[0]
        print(f"\n[类型3: 高Jaccard失败] route_id={case['route_id']}")
        print(f"  GT长度: {case['gt_len']}, 预测长度: {case['pred_len']}")
        print(f"  发散步: {case['diverge_step']}, 前缀匹配: {case['prefix_match_len']}")
        print(f"  Jaccard: {case['jaccard']:.3f}")
        print(f"  失败类型: {'hit_wall' if case['hit_wall'] else 'dead_end'}")
    
    # 案例4: rejoin_gt但仍失败
    rejoin_fail = [f for f in failures if f['rejoin_gt_after_diverge']]
    if rejoin_fail:
        case = rejoin_fail[0]
        print(f"\n[类型4: 重新汇入GT但失败] route_id={case['route_id']}")
        print(f"  GT长度: {case['gt_len']}, 预测长度: {case['pred_len']}")
        print(f"  发散步: {case['diverge_step']}, 前缀匹配: {case['prefix_match_len']}")
        print(f"  Jaccard: {case['jaccard']:.3f}")
        print(f"  rejoin_gt_after_diverge: {case['rejoin_gt_after_diverge']}")
    
    # 案例5: 发散后无法到达终点
    cannot_reach = [f for f in failures if not f['diverge_pred_can_reach_dest']]
    if cannot_reach:
        case = cannot_reach[0]
        print(f"\n[类型5: 发散后无法到达终点] route_id={case['route_id']}")
        print(f"  GT长度: {case['gt_len']}, 预测长度: {case['pred_len']}")
        print(f"  发散步: {case['diverge_step']}")
        print(f"  diverge_pred_can_reach_dest: False")
    else:
        print(f"\n[类型5: 发散后无法到达终点] - 无此类案例 (所有发散后均可达目的地)")

print("\n" + "=" * 80)
print("失败模式诊断总结")
print("=" * 80)

# 汇总两个城市的关键发现
all_failures = []
for city_data in report['per_city']:
    all_failures.extend(city_data['failures'])

total_failures = len(all_failures)
hit_wall_total = sum(1 for f in all_failures if f['hit_wall'])
dead_end_total = sum(1 for f in all_failures if f['dead_end'])

print(f"\n总失败案例: {total_failures}")
print(f"  - Hit Wall: {hit_wall_total} ({hit_wall_total/total_failures:.1%})")
print(f"  - Dead End: {dead_end_total} ({dead_end_total/total_failures:.1%})")

# 早期发散统计
early_diverge = sum(1 for f in all_failures if f['diverge_step'] <= 5)
print(f"\n早期发散 (step<=5): {early_diverge} ({early_diverge/total_failures:.1%})")

# 发散后仍可达
can_reach = sum(1 for f in all_failures if f['diverge_pred_can_reach_dest'])
print(f"发散后仍可达目的地: {can_reach} ({can_reach/total_failures:.1%})")

# 振荡模式
osc_count = 0
for f in all_failures:
    if f['hit_wall']:
        curs = [s['cur'] for s in f['last_k']['steps']]
        if len(set(curs)) == 2 and len(curs) >= 5:
            if curs[0] == curs[2] == curs[4]:
                osc_count += 1
print(f"Hit Wall中的振荡模式: {osc_count} ({osc_count/hit_wall_total:.1%} of hit_wall)")

print("\n【核心发现】")
print("1. 几乎所有失败案例在发散后仍可到达目的地 → 问题不在图的连通性")
print("2. 约78%的失败是hit_wall → 模型无法收敛到目的地，陷入循环")
print("3. 大量hit_wall案例展现振荡模式 → 在两个节点间来回跳动直到达到最大步数")
print("4. 发散通常发生在路线的早期 → 模型在初期就偏离了GT轨迹")
print("5. GT最终跳几乎总是在候选集中 → 图拓扑本身支持正确路径")

#!/usr/bin/env python3
"""对比 BASE vs CONTRAST 的退化机制"""
import json
import numpy as np

BASE_PATH = "_sync/wsa/oracle_step_runs/20260131_candq1_past8_contrast_vs_base_allcand_hop_attn_s0/base/oracle_step_diagnose_n200.json"
CONTRAST_PATH = "_sync/wsa/oracle_step_runs/20260131_candq1_past8_contrast_vs_base_allcand_hop_attn_s0/contrast/oracle_step_diagnose_n200.json"

with open(BASE_PATH) as f:
    base_data = json.load(f)

with open(CONTRAST_PATH) as f:
    contrast_data = json.load(f)

print("=" * 80)
print("BASE vs CONTRAST 对比分析")
print("=" * 80)

# 1. 总体性能
print("\n=== 1. 总体性能 ===")
base_succ = base_data["summary"]["success_rate"]
contrast_succ = contrast_data["summary"]["success_rate"]
print(f"Success rate: {base_succ:.4f} → {contrast_succ:.4f} (Δ={contrast_succ - base_succ:+.4f})")

base_exact = base_data["summary"]["success_exact_rate"]
contrast_exact = contrast_data["summary"]["success_exact_rate"]
print(f"Exact match rate: {base_exact:.4f} → {contrast_exact:.4f}")

base_div = base_data["summary"]["success_diverged_rate"]
contrast_div = contrast_data["summary"]["success_diverged_rate"]
print(f"Diverged success: {base_div:.4f} → {contrast_div:.4f}")

# 2. 分城市
print("\n=== 2. 分城市性能 ===")
base_city = base_data.get("summary_by_city", {})
contrast_city = contrast_data.get("summary_by_city", {})

for city_key in ["city0", "city1"]:
    base_sr = base_city.get(city_key, {}).get("true", {}).get("success_rate", 0)
    contrast_sr = contrast_city.get(city_key, {}).get("true", {}).get("success_rate", 0)
    city_name = "Detroit" if city_key == "city0" else "Columbus"
    print(f"{city_name}: {base_sr:.4f} → {contrast_sr:.4f} (Δ={contrast_sr - base_sr:+.4f})")

# 3. 偏离分析
print("\n=== 3. 偏离分析 ===")

base_routes = base_data["per_route"]
contrast_routes = contrast_data["per_route"]

# 统计divergence
base_div_count = sum(1 for r in base_routes if r["diverge_idx"] > 0)
contrast_div_count = sum(1 for r in contrast_routes if r["diverge_idx"] > 0)
print(f"总偏离数: {base_div_count} → {contrast_div_count}")

# 偏离步分布
base_div_steps = [r["diverge_idx"] for r in base_routes if r["diverge_idx"] > 0]
contrast_div_steps = [r["diverge_idx"] for r in contrast_routes if r["diverge_idx"] > 0]

print(f"偏离步中位数: {np.median(base_div_steps):.1f} → {np.median(contrast_div_steps):.1f}")
print(f"偏离步均值: {np.mean(base_div_steps):.1f} → {np.mean(contrast_div_steps):.1f}")

# 早期偏离占比
early_threshold = 5
base_early = sum(1 for s in base_div_steps if s <= early_threshold) / len(base_div_steps) if base_div_steps else 0
contrast_early = sum(1 for s in contrast_div_steps if s <= early_threshold) / len(contrast_div_steps) if contrast_div_steps else 0
print(f"早期偏离(≤{early_threshold}步)占比: {base_early:.3f} → {contrast_early:.3f}")

# 4. 首次偏离点分析
print("\n=== 4. 首次偏离点 GT 排名 ===")

def analyze_first_div(routes):
    ranks = []
    for r in routes:
        if "first_div_transition" in r and not r["success"]:
            trans = r["first_div_transition"]
            ranks.append(trans.get("gt_rank", 999))
    return ranks

base_ranks = analyze_first_div(base_routes)
contrast_ranks = analyze_first_div(contrast_routes)

print(f"失败样本数: {len(base_ranks)} → {len(contrast_ranks)}")
print(f"GT rank 中位数: {np.median(base_ranks):.1f} → {np.median(contrast_ranks):.1f}")
print(f"GT rank p95: {np.percentile(base_ranks, 95):.1f} → {np.percentile(contrast_ranks, 95):.1f}")
print(f"GT rank max: {np.max(base_ranks):.1f} → {np.max(contrast_ranks):.1f}")

# 5. Close-call 比例
print("\n=== 5. Close-Call 分析 ===")
base_cc = base_data["q2_logits"]["first_div_close_call_frac"]
contrast_cc = contrast_data["q2_logits"]["first_div_close_call_frac"]
print(f"Close-call 比例: {base_cc:.4f} → {contrast_cc:.4f}")

# 6. 超长游走（pred_len == max_decode_len+1）
print("\n=== 6. 超长游走分析 ===")

def count_hit_wall(routes):
    max_len = 161  # max_decode_len + 1
    count = sum(1 for r in routes if not r["success"] and r["pred_len"] == max_len)
    fail_count = sum(1 for r in routes if not r["success"])
    return count, fail_count, count / fail_count if fail_count > 0 else 0

base_hw, base_fail, base_hw_pct = count_hit_wall(base_routes)
contrast_hw, contrast_fail, contrast_hw_pct = count_hit_wall(contrast_routes)

print(f"失败样本中超长游走: {base_hw}/{base_fail} ({base_hw_pct:.3f}) → {contrast_hw}/{contrast_fail} ({contrast_hw_pct:.3f})")

# 7. Logit margin 变化
print("\n=== 7. Logit Margin 变化 ===")
base_margin_p50 = base_data["q2_logits"]["first_div_logit_margin_quantiles"]["p50"]
contrast_margin_p50 = contrast_data["q2_logits"]["first_div_logit_margin_quantiles"]["p50"]
print(f"Logit margin (p50): {base_margin_p50:.4f} → {contrast_margin_p50:.4f}")

# 8. 候选选择逻辑（pred_closer_to_dest）
print("\n=== 8. 距离偏好分析 ===")
base_closer = base_data["q4_dest_dist_shortcut"]["first_div_pred_closer_to_dest_frac"]
contrast_closer = contrast_data["q4_dest_dist_shortcut"]["first_div_pred_closer_to_dest_frac"]
print(f"pred 更靠近终点比例: {base_closer:.4f} → {contrast_closer:.4f}")

print("\n" + "=" * 80)
print("结论总结")
print("=" * 80)
print(f"""
1. 性能严重下降：-24.75pp（82.25% → 57.5%）
   - Detroit 更严重：-31.5pp（79.5% → 48%）
   - Columbus 也下降：-18.0pp（85% → 67%）

2. 退化机制：更早且更多的偏离
   - 偏离数增加：{base_div_count} → {contrast_div_count}
   - 早期偏离占比增加：{base_early:.1%} → {contrast_early:.1%}
   - 首次偏离点 GT rank 恶化：p95 2→3, max 5→6

3. 不是"不确定"而是"自信地选错"
   - Close-call 下降：28.7% → 18.6%（反而更自信）
   - 模型确信选了错的候选

4. 模型学到了有害的选择偏好
   - diff_from_mean 特征被学成了负面特征
   - 或者权重初始化方式导致 contrast 项压倒了原有的正确决策

建议：停止 contrast 方向，不值得继续迭代。
""")

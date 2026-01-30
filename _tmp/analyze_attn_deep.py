#!/usr/bin/env python3
"""深度分析 candq1 失败案例的 attention 特征"""
import json
import numpy as np

DATA_PATH = "_sync/wsa/icml2026_routegen/WAYCASD_AB_candq1_pastctx_k8_strict_sem5_seed0_e100/W8_diag/oracle_step_n200_attn.json"

with open(DATA_PATH) as f:
    d = json.load(f)

failures = [r for r in d["per_route"] if not r["success"]]
print(f"Total failures: {len(failures)}")

# 计算 attention 相似度统计
cos_list = []
top1_same = 0
pred_entropy_list = []
gt_entropy_list = []
ctx_diff_list = []
cand_h_diff_list = []
logit_gap_list = []

def entropy(p):
    p = np.clip(p, 1e-9, 1)
    return -np.sum(p * np.log(p))

for r in failures:
    trans = r["first_div_transition"]
    pred_attn = np.array(trans["pred_attn_weight"])
    gt_attn = np.array(trans["gt_attn_weight"])
    
    # cosine similarity
    cos = np.dot(pred_attn, gt_attn) / (np.linalg.norm(pred_attn) * np.linalg.norm(gt_attn) + 1e-9)
    cos_list.append(cos)
    
    # top1 same
    if np.argmax(pred_attn) == np.argmax(gt_attn):
        top1_same += 1
    
    pred_entropy_list.append(entropy(pred_attn))
    gt_entropy_list.append(entropy(gt_attn))
    
    # 其他特征
    ctx_diff_list.append(trans.get("ctx_diff_norm", 0))
    cand_h_diff_list.append(trans.get("cand_h_diff", 0))
    logit_gap_list.append(trans.get("gt_gap", 0))

cos_arr = np.array(cos_list)
print(f"\n=== Attention Similarity (pred vs gt) ===")
print(f"cos(pred_attn, gt_attn):")
print(f"  mean={cos_arr.mean():.4f}, median={np.median(cos_arr):.4f}")
print(f"  p10={np.percentile(cos_arr, 10):.4f}, p90={np.percentile(cos_arr, 90):.4f}")
print(f"  min={cos_arr.min():.4f}, max={cos_arr.max():.4f}")

print(f"\ntop1_same_frac: {top1_same}/{len(failures)} = {top1_same/len(failures):.3f}")

n_high_cos = (cos_arr >= 0.95).sum()
n_low_cos = (cos_arr <= 0.50).sum()
print(f"\ncos >= 0.95: {n_high_cos}/{len(failures)} = {n_high_cos/len(failures):.3f}")
print(f"cos <= 0.50: {n_low_cos}/{len(failures)} = {n_low_cos/len(failures):.3f}")

print(f"\n=== Entropy ===")
print(f"entropy pred: mean={np.mean(pred_entropy_list):.3f}")
print(f"entropy gt: mean={np.mean(gt_entropy_list):.3f}")

print(f"\n=== Context & Embedding Diff ===")
print(f"ctx_diff_norm: mean={np.mean(ctx_diff_list):.4f}, median={np.median(ctx_diff_list):.4f}")
print(f"cand_h_diff: mean={np.mean(cand_h_diff_list):.4f}, median={np.median(cand_h_diff_list):.4f}")
print(f"logit_gap (gt_gap): mean={np.mean(logit_gap_list):.4f}, median={np.median(logit_gap_list):.4f}")

# 分组分析: cos >= 0.95 vs cos < 0.95
print(f"\n=== 分组分析: cos >= 0.95 (attention几乎相同) ===")
high_cos_idx = cos_arr >= 0.95
low_cos_idx = cos_arr < 0.95

high_cos_logit_gap = [logit_gap_list[i] for i in range(len(failures)) if high_cos_idx[i]]
low_cos_logit_gap = [logit_gap_list[i] for i in range(len(failures)) if low_cos_idx[i]]

high_cos_cand_diff = [cand_h_diff_list[i] for i in range(len(failures)) if high_cos_idx[i]]
low_cos_cand_diff = [cand_h_diff_list[i] for i in range(len(failures)) if low_cos_idx[i]]

print(f"High cos (>=0.95, n={len(high_cos_logit_gap)}):")
print(f"  logit_gap: mean={np.mean(high_cos_logit_gap):.4f}, median={np.median(high_cos_logit_gap):.4f}")
print(f"  cand_h_diff: mean={np.mean(high_cos_cand_diff):.4f}, median={np.median(high_cos_cand_diff):.4f}")

print(f"Low cos (<0.95, n={len(low_cos_logit_gap)}):")
print(f"  logit_gap: mean={np.mean(low_cos_logit_gap):.4f}, median={np.median(low_cos_logit_gap):.4f}")
print(f"  cand_h_diff: mean={np.mean(low_cos_cand_diff):.4f}, median={np.median(low_cos_cand_diff):.4f}")

# 检查 close_call 比例
close_call_high = sum(1 for i, r in enumerate(failures) if high_cos_idx[i] and r["first_div_transition"].get("close_call", False))
close_call_low = sum(1 for i, r in enumerate(failures) if low_cos_idx[i] and r["first_div_transition"].get("close_call", False))

print(f"\nclose_call in high_cos: {close_call_high}/{len(high_cos_logit_gap)} = {close_call_high/len(high_cos_logit_gap):.3f}")
print(f"close_call in low_cos: {close_call_low}/{len(low_cos_logit_gap):.3f}" if low_cos_logit_gap else "close_call in low_cos: N/A")

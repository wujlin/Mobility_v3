# Way-CASD 实验计划 (2026-02-04)

---

## 已验证的核心洞见

| 洞见 | Evidence | 结论 |
|------|----------|------|
| **past_k=16 有效** | Oracle [60,+): 76%→85.3% (+9pp), loop: 30%→14.7% | ✅ 历史窗口是真瓶颈 |
| **Flow-compat 可用** | 旧Flow+新AE: [60,+)=44.1% vs baseline 38.2% (+6pp) | ✅ past16 decoder 本身更强 |
| **K>4 无效** | K=6降到23.5%, K=8降到35.3% | ❌ K=4已是最优 |
| **RL对AR敏感** | GT下44.1%, AR下32.4% (gap=11.7pp) | ⚠️ 需改进训练方式 |
| **Multiscale失败** | Oracle从85%降到47% | ❌ 当前架构不可用 |

---

## 下一步实验计划（按优先级排序）

### 🥇 实验1：E5 Flow 正确配置重训

**背景**：E5 Flow-retrain v2 失败是因为配置错误（`region_seq_npz=null`），不是past_k的问题。E5 Flow-compat已经证明past16 decoder有效（+6pp）。

**核心假设**：正确配置的Flow + past16 AE 应该能进一步提升。

**行动**：
```bash
# 重训 Flow，确保以下配置正确
python src/training/train_way_casd_flow.py \
    --ae_ckpt W9_train_ae_min5_candq1_past16_len160_s0/ckpt_best.pt \
    --use_region_seq \
    --region_seq_npz <与W10相同的路径> \
    --way_regions_npz <与W10相同的路径> \
    --n_layers 6 \
    --cond_inject xattn \
    --out_dir W10_train_flow_past16_regionseq_xattn_s0
```

**检查点**：
1. 训练前：确认 `report.json` 中 `region_seq_npz != null`
2. 训练后：评测 `binned_E5_flow_v3.json`

**预期结果**：
- [60,+) success: 45-50%（比Flow-compat的44.1%略高，因为latent和decoder匹配）
- 如果低于44%，说明有其他问题需要排查

**验收标准**：[60,+) success ≥ 45%

---

### 🥈 实验2：E7 RL v2 — Region扰动训练

**背景**：E7 RL在GT region下有效（44.1%），但在AR region下差（32.4%）。Gap=11.7pp说明RL过拟合了GT region。

**核心假设**：如果训练时混入AR region或加噪声，decoder应该学会容忍region误差。

**行动**：

**方案A（推荐）：混合训练**
```python
# 修改 RL 训练代码
def get_region_constraint(batch):
    if random.random() < 0.5:
        return batch["gt_region_seq"]      # 50% 用 GT
    else:
        return region_ar.sample(batch)     # 50% 用 AR 采样
```

**实现提示（当前代码接口）**：
- `src/training/train_way_casd_decoder_rl.py` 已支持：
  - `--region_constraint mix --region_mix_gt_prob 0.5 --region_ar_ckpt <ckpt>`
  - `--region_noise_p 0.15`（train-only，可与 gt/mix 组合）

**方案B：Region噪声注入**
```python
def perturb_region(region_seq, adj_matrix, p=0.15):
    """以概率p将region替换为相邻region"""
    perturbed = region_seq.clone()
    for t in range(len(region_seq)):
        if random.random() < p:
            neighbors = adj_matrix[region_seq[t]].nonzero().squeeze(-1)
            if len(neighbors) > 0:
                perturbed[t] = neighbors[torch.randint(len(neighbors), (1,))]
    return perturbed
```

**检查点**：
1. 使用实验1的最佳Flow checkpoint
2. 同时评测 GT region 和 AR region 两个口径

**预期结果**：
- AR region [60,+): 38-42%（从32.4%提升6-10pp）
- GT region [60,+): 维持40%+（不应退化太多）
- Gap缩小到5pp以内

**验收标准**：AR region [60,+) success ≥ 38%，GT/AR gap ≤ 6pp

---

### 🥉 实验3：past_k=24 验证（可选）

**背景**：past_k=16比past_k=8提升了9pp。如果继续增大，是否还有收益？

**核心假设**：past_k=24可能进一步提升Oracle上限，但边际收益可能递减。

**行动**：
```bash
# 只训练AE，验证Oracle上限
python src/training/train_way_casd_ae.py \
    --decoder_past_k 24 \
    --out_dir W9_train_ae_min5_candq1_past24_len160_s0
```

**检查点**：
1. 只做Oracle评测（不需要训练Flow）
2. 对比 past_k=8/16/24 的Oracle曲线

**预期结果**：
- 如果 [60,+) Oracle > 88%：说明还有空间，值得继续
- 如果 [60,+) Oracle ≈ 85%：说明已饱和，不再增大

**验收标准**：Oracle [60,+) success > 87%才值得后续投入

---

## 实验执行顺序

```
Day 1-2: 实验1（Flow重训）
    └── 训练完成后立即评测
    
Day 2-3: 实验2（RL v2）
    └── 基于实验1的checkpoint
    └── 需要修改训练代码
    
Day 3（可选）: 实验3（past_k=24）
    └── 只是验证上限，优先级低
```

---

## 当前最佳配置（Baseline for comparison）

| 组件 | 配置 | 来源 |
|------|------|------|
| AE | past_k=16, cand_query=True | E5 |
| Flow | RegionSeq xattn (旧，待重训) | W10 |
| Decode | beam=10, soft P=2.0, K=4 | E4 |
| Region | ar, relaxed, dest_region fallback | 已验证 |

**当前[60,+) success: 44.1%**（E5 Flow-compat）

**目标：通过实验1+2达到 50%+**

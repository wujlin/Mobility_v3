# Way-CASD 实验进展报告 (2026-02-04 更新)

---

## 📊 最新实验结果总览

### 核心指标对比（Beam, [60,+) success）

| 实验 | 配置 | [60,+) | hit_wall | loop | vs 基准 |
|------|------|--------|----------|------|---------|
| **E5 past16 Oracle** | past_k=16, GT latent | **85.3%** | 14.7% | 14.7% | Oracle上界 |
| **E3 past24 Oracle** | past_k=24, GT latent | 55.9% | 44.1% | 41.2% | **-29.4pp** 🚨 |
| **E1 Flow v3** | past16 + Flow(RegionSeq) | **50.0%** | 50.0% | 50.0% | **+5.9pp** ✅ |
| E5 Flow-compat | 旧Flow + 新past16 AE | 44.1% | - | - | 上一轮最佳 |
| E2 RL v2 (AR) | RL + Flow, AR约束 | 38.2% | 61.8% | 47.1% | **-11.8pp** ❌ |
| E2 RL v2 (GT) | RL + Flow, GT约束 | 32.4% | 64.7% | 58.8% | **-17.6pp** ❌ |

---

## 🔬 Non-trivial Insights

### 1. E1 Flow v3 达到50%——Latent匹配的价值

E1（past16 AE + 正确配置的Flow with RegionSeq xattn）达到**50.0%** [60,+)：
- 比Flow-compat（44.1%）提升**+5.9pp**
- 超过预设验收标准（≥45%）

**洞见**：Flow与Decoder的latent分布匹配至关重要。旧Flow给新Decoder存在分布偏移，重训后匹配，效果更佳。

### 2. past_k=24显著退步——Context Window存在甜点区 🚨

past24 Oracle的[60,+)只有**55.9%**，远低于past16的**85.3%**（-29.4pp）。

**分析**：
- 更长的history窗口可能引入**无关噪声**（早期点与当前决策无关）
- Transformer attention在更长序列上**稀释**，关键位置信号减弱
- 24步context的组合空间更大，可能需要更深网络或更多数据

**结论**：past_k=16是当前架构下的最优选择，**不建议进一步增加**。

### 3. RL在GT约束下反而更差——Region引导的悖论 ⚠️

| 约束 | [60,+) | 短桶[5,10) |
|------|--------|-----------|
| AR | 38.2% | 60.4% |
| GT | 32.4% | 75.0% |

GT在短桶更好，但**长桶更差**。

**推测解释**：
- 长程生成中一旦偏离GT region，后续GT约束变成**干扰而非引导**
- RL的region mix训练让decoder学会在不确定时"保守前进"，AR下反而受益
- 对长程生成，"模糊可调"的约束比"精确不容错"的约束**更鲁棒**

### 4. RL当前设定无效——MLE仍是最优

| 方法 | [60,+) | hit_wall | loop |
|------|--------|----------|------|
| E1 MLE | 50.0% | 50.0% | 50.0% |
| E2 RL | 38.2% | 61.8% | 47.1% |

RL让hit_wall恶化（+11.8pp），虽然loop略有改善（-2.9pp），但总体差11.8pp。

**可能原因**：
- Reward设计过于强调reach-dest而忽略path quality
- Region mix引入额外不确定性，有限样本下难以收敛

### 5. 城市差异显著——Columbus >> Detroit

E1 per-city [60,+)：
- Detroit: 40.9%
- Columbus: **66.7%**（+25.8pp）

**启示**：Columbus路网更规则，长程规划更易；论文应分城市报告。

---

## 已验证的核心洞见（完整版）

| 洞见 | Evidence | 结论 |
|------|----------|------|
| **past_k=16 有效** | Oracle [60,+): 76%→85.3% (+9pp) | ✅ 历史窗口是真瓶颈 |
| **past_k=24 退步** | Oracle [60,+): 85.3%→55.9% (-29.4pp) | ❌ 存在甜点区，16是最优 |
| **Flow重训有效** | [60,+): 44.1%→50.0% (+5.9pp) | ✅ Latent匹配重要 |
| **K>4 无效** | K=6降到23.5%, K=8降到35.3% | ❌ K=4已是最优 |
| **RL当前无效** | MLE 50% vs RL 38.2% (-11.8pp) | ❌ 需改进reward设计 |
| **GT约束长程悖论** | GT 32.4% < AR 38.2% | ⚠️ 长程需要容错约束 |

---

## 下一步实验建议（基于当前发现）

### 🥇 方向1：攻克hit_wall（优先级最高）

**问题**：E1的[60,+) hit_wall仍有50%，是主要瓶颈。

**建议实验**：
```bash
# A) Guided Dest Alpha = 0.1~0.3
python src/evaluation/way_casd_binned_eval.py \
    --decode_guided_dest_alpha 0.2 \
    --ae_ckpt W9 --flow_ckpt W11 ...

# B) Past Context更深：n_layers=4
# 需要重训AE：--decoder_past_n_layers=4（past_k保持16）

# C) 候选扩展：decode_max_candidates=5
python src/evaluation/way_casd_binned_eval.py \
    --decode_max_candidates 5 \
    --ae_ckpt W9 --flow_ckpt W11 ...
```

**预期**：hit_wall从50%降到35-40%，success提升到55-60%。

---

### 🥈 方向2：RL Reward重设计

**问题**：当前RL让hit_wall恶化（+11.8pp），reward设计有问题。

**建议改进**：
1. 增加`path_smoothness_reward`：惩罚急转弯
2. 增加`wall_proximity_penalty`：接近max_len时惩罚
3. 先在纯GT region下调通，再加region mix

（实现提示：`src/training/train_way_casd_decoder_rl.py` 已支持 `--penalty_turn/--penalty_hit_wall/--penalty_wall/--wall_margin`，可用于快速验证 reward redesign。）

---

### 🥉 方向3：分城市分析（论文价值）

Columbus显著优于Detroit（66.7% vs 40.9%）。

**建议分析**：
1. 路网dead-end比例对比
2. Region划分粒度差异
3. 训练数据分布

（实现提示：`src/evaluation/way_casd_city_data_audit.py` 已支持 `--way_regions_npz`，输出 dead_end_frac 与 region_size 分布。）

**论文可作为：** "城市拓扑结构对长程路由生成的影响" 小节。

---

## 当前最佳配置

| 组件 | 配置 | 来源 |
|------|------|------|
| **AE** | past_k=16, cand_query=True, len=160 | W9/E5 |
| **Flow** | RegionSeq xattn, n_layers=6 | **W11 (新!)** |
| **Decode** | beam=10, soft P=2.0, K=4 | E4 |
| **Region** | AR, relaxed, dest_region fallback | 已验证 |

**当前最佳 [60,+) success: 50.0%**（E1 Flow v3）

---

## Checkpoints索引

| 实验 | AE ckpt | Flow ckpt | Decoder ckpt |
|------|---------|-----------|--------------|
| E1 Flow v3 | W9 (past16) | **W11 (RegionSeq)** | same as AE |
| E2 RL v2 | W12 (RL fine-tuned) | W11 | W12 |
| E3 past24 | W13 (past24) | - | same as AE |
| Baseline | W9 | W10 (旧) | same as AE |

---

## 结论

1. **E1 Flow v3是当前最优配置**，[60,+) success=50.0%，比上轮+5.9pp
2. **past_k=16是最优窗口**，24会显著退步（-29.4pp）
3. **RL在当前设定下无效**（-11.8pp），需要redesign reward
4. **GT约束长程反而不如AR**，说明长程需要容错机制
5. **Columbus >> Detroit**（66.7% vs 40.9%），城市拓扑影响显著

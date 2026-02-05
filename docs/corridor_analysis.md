# Corridor Diversity 叙事主线

> 目标：为CascadeTraj论文构建清晰的corridor diversity叙事
> 更新日期：2026-02-05

---

## 一、叙事主线（Problem → Insight → Method → Contribution）

### 1. Problem: Route ≠ Shortest Path，但现有方法不知道"远离shortest path多少"

**现实观察**：
- 同一OD对，人们会选择多条不同的路线
- 这些路线在空间上形成**若干"走廊"（corridors）**
- 不同corridor反映不同的出行偏好（避堵、避风险、scenic route等）

**现有方法的缺陷**：
- Shortest path：完全忽略corridor diversity
- Learned cost + search (GTG)：只能学到"平均偏好"，输出单一路径
- Diffusion in GPS space：生成多样性是"噪声多样性"而非"corridor多样性"
- 纯AR序列生成：缺乏对corridor结构的建模

**核心问题**：
> 如何生成**在corridor层面真正多样**的路线，而非仅在细节上抖动？

### 2. Insight: Corridor = Coarse Structure, Route = Fine Execution

**关键洞见**：
- 人的路线选择是**层次化**的：
  1. **Corridor选择**（走高速还是走市区？走南边还是北边？）
  2. **Route执行**（在选定corridor内，具体走哪些街道）

- 这种层次化结构在现有方法中**完全缺失**

**我们的建模**：
```
Region Sequence ≈ Corridor (coarse)
Way Sequence = Route (fine)
```

### 3. Method: Hierarchical Generation with Explicit Corridor Modeling

**CascadeTraj架构**：
1. **Corridor生成**：Region-level AR model预测corridor skeleton
2. **Latent空间**：Flow Matching生成route latent（条件于corridor）
3. **Route解码**：Graph-constrained AR decoder生成way sequence

**为什么这样设计能捕捉corridor diversity？**
- Region AR天然建模"选哪条走廊"
- Flow Matching在**同一corridor内**生成多样的route latent
- Decoder保证拓扑可行性

### 4. Contribution: Corridor-Aware Metrics + Controllable Generation

**方法贡献**：
- 首个显式建模corridor的route generation框架
- Region hierarchy作为corridor的proxy

**评估贡献**：
- 提出corridor-aware evaluation metrics
- 不仅看"路线是否正确"，还看"corridor分布是否正确"

**应用贡献**：
- 可控生成：指定corridor，生成该corridor内的多样路线
- 流量估计：正确估计各corridor的流量份额

---

## 二、Corridor Diversity 的量化定义

### 2.1 从文献中借鉴的框架

**Corridor Diversity = Separation × Effective Number**

| 维度 | 定义 | 文献来源 |
|------|------|----------|
| **Separation** | 不同corridor之间的距离 | APD (ML), CF/PS (transport) |
| **Effective Number** | 有多少个真正不同的corridor | Entropy, k-paths (VLDB) |

### 2.2 我们的可操作定义

```python
# Step 1: 定义corridor
corridor = cluster of routes sharing similar region_seq

# Step 2: Separation (corridor间距离)
separation = mean pairwise LCS distance between corridor prototypes

# Step 3: Effective Number (均衡性)
effective_k = exp(entropy of corridor traffic share)

# Step 4: Combined metric
corridor_diversity = separation × log(effective_k)
```

### 2.3 为什么用LCS而非Hausdorff？

| 度量 | 优点 | 缺点 |
|------|------|------|
| Hausdorff | 不需要map matching | 对平行路/立交敏感 |
| LCS (way seq) | 直接度量"走了哪条路" | 需要map matching |
| **我们选择LCS** | 因为我们的模型直接在way sequence上生成 | |

---

## 三、实验设计：证明Corridor Diversity

### 3.1 RQ1: 现有方法有corridor diversity吗？

**实验**：对比各方法在同一OD上生成K=10条路线

| Method | Sample内APD | Corridor数 | Effective K |
|--------|-------------|-----------|-------------|
| Shortest Path | 0 | 1 | 1 |
| GTG | 0 | 1 | 1 |
| DiffTraj | ~high | ~1-2 | ~1-2 |
| **CascadeTraj** | **high** | **3-5** | **3-4** |

**预期结论**：只有CascadeTraj能生成真正不同的corridors

### 3.2 RQ2: Corridor分布是否与GT匹配？

**实验**：对比生成的corridor份额 vs GT数据中的corridor份额

| Method | KL(gen || GT) | Corridor Coverage |
|--------|---------------|-------------------|
| DiffTraj | high | low |
| **CascadeTraj** | **low** | **high** |

### 3.3 RQ3: Corridor分布匹配度

**实验**：对比生成的corridor份额分布 vs GT数据中的分布

| Method | KL(gen || GT) | JS Divergence | Top-1 Corridor Match |
|--------|---------------|---------------|---------------------|
| Shortest Path | - | - | low |
| DiffTraj | high | high | medium |
| **CascadeTraj** | **low** | **low** | **high** |

**预期结论**：CascadeTraj能自动生成与GT分布匹配的corridor，无需人为指定

---

## 四、Introduction重写建议

### 当前问题
- "corridor"突然出现，没有铺垫
- motivation是"previous works fail..."（防御式）
- 没有数据支撑的claim

### 建议结构

```
P1: Route ≠ Shortest Path (data evidence)
    - "In Detroit, only 23% of observed trips follow shortest path"
    - "Same OD, multiple distinct corridors exist"

P2: Why corridor matters
    - Traffic planning needs corridor-level flow estimation
    - Simulation needs corridor diversity for realism
    
P3: Current methods fail to capture corridor
    - Shortest path: single output
    - Learned cost: single output (average preference)
    - Diffusion: "noisy diversity" ≠ corridor diversity
    
P4: Our insight: Hierarchical decision
    - Corridor choice (which main roads?)
    - Route execution (which streets within corridor?)
    
P5: Our method: CascadeTraj
    - Region-level hierarchy models corridor
    - Flow + AR decoder generates diverse routes
    
P6: Contributions (assertive)
    - First corridor-aware route generation framework
    - Novel corridor diversity metrics
    - State-of-the-art on [60,+) route success rate
```

---

## 五、Figure 1 建议（论文opening figure）

**一张图说清楚corridor diversity**：

```
[Left] GT routes from Detroit (same OD)
       → 3 visible corridors (north, central, south)
       → Color-coded by corridor

[Middle] Existing methods
       → Shortest path: 1 line
       → GTG: 1 line (slightly different)
       → DiffTraj: 10 lines but all in same corridor

[Right] CascadeTraj (ours)
       → 10 lines spanning all 3 corridors
       → Correctly matches corridor distribution
```

---

## 六、待验证的Claims（需要实验数据支撑）

1. **"23% of trips follow shortest path"** → 需要从数据计算
2. **"Existing methods collapse to 1-2 corridors"** → 需要baseline实验
3. **"CascadeTraj correctly estimates corridor flow share"** → 需要KL实验

---

## Partner执行指南

### 数据分析任务
```bash
# 1. 计算shortest path match rate
python src/evaluation/shortest_path_analysis.py \
  --way_routes_npz ... \
  --way_graph_npz ... \
  --output corridor_analysis/shortest_match.json

# 2. 提取GT corridor分布
python src/evaluation/corridor_extraction.py \
  --way_routes_npz ... \
  --method lcs_cluster \
  --n_clusters 5 \
  --output corridor_analysis/gt_corridors.json
```

### Baseline corridor评估
```bash
# 对每个baseline，生成K=10 samples per OD，计算corridor metrics
python src/evaluation/corridor_diversity_eval.py \
  --method {shortest_path/gtg/difftraj/ours} \
  --k_samples 10 \
  --output corridor_analysis/{method}_corridor.json
```

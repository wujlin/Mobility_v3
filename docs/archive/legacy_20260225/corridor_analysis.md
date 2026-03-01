# Corridor Diversity 叙事主线

> 目标：为CascadeTraj论文构建清晰的corridor diversity叙事
> 更新日期：2026-02-05
> 状态：已审核，删除循环论证

---

## 一、叙事主线（Problem → Hypothesis → Method → Validation）

### 1. Problem: Route ≠ Shortest Path

**可验证的观察**：
- 同一OD对，人们选择多条不同的路线
- 这些路线在空间上形成若干**走廊（corridors）**
- 需要从数据验证：shortest path match rate是多少？

**现有方法的局限**：
- Shortest path / learned cost：输出单一路径
- Diffusion in GPS space：多样性来源不明确（噪声？还是真正的route choice？）

### 2. Hypothesis（不是Insight）: 路线选择是层次化的

**我们的建模假设**（需要验证，不是既定事实）：
- 人的路线选择可能是层次化的：
  1. 先选corridor（走北边还是南边？）
  2. 再选具体route（哪条街道？）

**我们的proxy**：
- 用region sequence近似corridor
- **待验证**：region_seq是否是corridor的有效proxy？

### 3. Method: Region-Conditioned Generation

**CascadeTraj架构**：
- Region AR：预测region sequence（corridor的proxy）
- Flow Matching：生成route latent（条件于region）
- AR Decoder：生成way sequence（graph约束）

### 4. Validation: 需要回答的问题

1. **Region是否是corridor的好proxy？**
   - 方法：先用way sequence LCS聚类定义"GT corridor"
   - 验证：同一GT corridor内的routes是否有相似的region_seq？

2. **各方法的corridor分布是否与GT匹配？**
   - 用独立定义的corridor（LCS聚类）评估所有方法
   - 比较KL divergence

---

## 二、Corridor的独立定义（避免循环论证）

### 关键原则
> Corridor的定义**必须独立于**我们的region_seq，否则是循环论证

### 2.1 GT Corridor定义（基于way sequence）

```python
# Step 1: 对同一OD的所有GT routes，计算pairwise LCS similarity
def lcs_similarity(route_a, route_b):
    lcs_len = longest_common_subsequence(route_a, route_b)
    return lcs_len / min(len(route_a), len(route_b))

# Step 2: 聚类得到corridors
# 使用层次聚类或DBSCAN，threshold根据数据调整
corridors = hierarchical_cluster(routes, similarity=lcs_similarity, threshold=0.5)
```

### 2.2 验证Region是否是好的Corridor Proxy

```python
# 如果region_seq是好的corridor proxy，那么：
# 同一corridor内的routes应该有相似的region_seq

for corridor in gt_corridors:
    region_seqs = [get_region_seq(route) for route in corridor.routes]
    intra_corridor_region_similarity = mean_pairwise_similarity(region_seqs)
    # 应该接近1.0
```

---

## 三、实验设计（去除预设结论）

### EXP1: Shortest Path Match Rate
**问题**：GT数据中多少比例的路线与shortest path一致？
**意义**：如果比例很低，证明问题non-trivial

### EXP2: GT Corridor提取
**问题**：同一OD的GT routes能聚类成多少个corridors？
**方法**：LCS聚类
**意义**：建立corridor diversity的ground truth

### EXP3: Region作为Corridor Proxy的有效性
**问题**：region_seq能否区分不同的GT corridors？
**方法**：计算corridor内vs corridor间的region similarity
**预期**：如果有效，intra-corridor similarity >> inter-corridor similarity

### EXP4: 各方法的Corridor分布比较
**问题**：各方法生成的routes的corridor分布与GT是否匹配？
**方法**：
1. 对每个OD，用GT corridors作为cluster centers
2. 将生成的routes分配到最近的corridor
3. 比较分布

| Method | Corridor Coverage | KL(gen \|\| GT) |
|--------|------------------|-----------------|
| Shortest Path | ? | ? |
| RNN AR | ? | ? |
| DiffTraj | ? | ? |
| CascadeTraj | ? | ? |

---

## 四、Introduction结构（修正版）

```
P1: Route ≠ Shortest Path (data evidence)
    - "In Detroit, X% of observed trips deviate from shortest path"
    - 具体数字来自EXP1

P2: Corridor matters for planning
    - Traffic planning needs corridor-level flow
    - Simulation needs corridor diversity

P3: Current methods' limitation
    - Single-output methods: cannot capture diversity
    - Diffusion methods: diversity source unclear
    
P4: Our hypothesis: Hierarchical route choice
    - Corridor (coarse) + Route (fine)
    - Region as corridor proxy（诚实说是hypothesis）
    
P5: CascadeTraj
    - Region-conditioned hierarchical generation
    
P6: Contributions
    - Hierarchical route generation framework
    - Corridor-aware evaluation protocol
    - Empirical validation on Detroit/Columbus
```

---

## 五、Figure 1（需要真实数据）

**展示corridor diversity的直观证据**：
- 选一个OD，展示GT routes按corridor着色
- 对比各方法的生成结果

**注意**：必须用EXP2的结果，不能人为画

---

## 六、待完成的数据分析

| 任务 | 状态 | 负责 |
|------|------|------|
| EXP1: Shortest path match rate | ⬜ | Partner |
| EXP2: GT corridor提取 (LCS聚类) | ⬜ | Partner |
| EXP3: Region-corridor correspondence | ⬜ | Partner |
| EXP4: 各方法corridor比较 | ⬜ 依赖baseline | Partner |

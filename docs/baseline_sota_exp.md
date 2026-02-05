# Baseline & SOTA 比较实验设计

> 目标：为NeurIPS 2026论文准备完整的comparison实验
> 更新日期：2026-02-05

---

## 实验分类

| 类型 | 模型 | 目的 | 可并行 |
|------|------|------|--------|
| **Baseline** | Shortest Path | 证明问题non-trivial | ✅ |
| **Baseline** | RNN AR | 序列生成经典方法 | ✅ |
| **Baseline** | Transformer AR | 更强的AR baseline | ✅ |
| **SOTA** | GTG (AAAI 2025) | 学cost+search方法 | ✅ |
| **SOTA** | DiffTraj/Cardiff | Diffusion方法 | ✅ |
| **Ablation** | Way-CASD (no region) | 证明hierarchy价值 | ✅ |
| **Ablation** | Way-CASD (no past_k) | 证明past context价值 | ✅ |

**所有模型可并行训练，不依赖我们的模型改进。**

---

## 一、Baselines (简单方法)

### B1: Shortest Path ⬜ 待执行
**目的**：最基础baseline，证明人不走最短路

**方法**：
```python
# 给定OD pair，直接返回shortest path
def shortest_path_baseline(G, start, dest):
    return nx.shortest_path(G, start, dest, weight='length')
```

**实现**：
- 直接用way_graph的邻接关系
- 不需要训练，只需要评估

**评估**：
- Success: 如果shortest path存在且能到达dest → 100%
- 关键指标：DTW/Fréchet与GT的差异（应该很大）

**输出**：`_sync/wsa/baselines/B1_shortest_path/`

---

### B2: RNN AR ⬜ 待执行
**目的**：经典序列生成baseline

**架构**：
```python
class RNNARDecoder(nn.Module):
    def __init__(self, d_model=256, n_layers=2):
        self.way_emb = nn.Embedding(n_ways, d_model)
        self.cond_enc = ConditionEncoder(...)  # OD, time
        self.rnn = nn.GRU(d_model, d_model, n_layers, batch_first=True)
        self.scorer = nn.Linear(d_model, 1)  # score candidates
    
    def forward(self, way_seq, candidates, cond):
        # Teacher forcing: 给定前缀，预测下一个way
        h = self.cond_enc(cond)  # initial hidden
        for t in range(len(way_seq)):
            emb = self.way_emb(way_seq[t])
            out, h = self.rnn(emb, h)
            scores = self.scorer(out)  # score over candidates
```

**训练**：
- Teacher forcing with CE loss
- 使用相同的candidate mask（公平比较）

**输出**：`_sync/wsa/baselines/B2_rnn_ar/`

---

### B3: Transformer AR ⬜ 待执行
**目的**：更强的AR baseline（无latent）

**架构**：
```python
class TransformerARDecoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8):
        self.way_emb = nn.Embedding(n_ways, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.cond_enc = ConditionEncoder(...)
        self.transformer = nn.TransformerDecoder(...)
        self.scorer = nn.Linear(d_model, 1)
```

**与Way-CASD的关键区别**：
- 无latent compression（直接AR）
- 无hierarchy（无region约束）

**输出**：`_sync/wsa/baselines/B3_transformer_ar/`

---

## 二、SOTA Methods (需要复现)

### S1: GTG (AAAI 2025) ⬜ 待执行
**论文**：Graph Trajectory Generation with Learning-based Cost

**核心思路**：
- 学习edge cost function
- 用learned cost做graph search (A*/Dijkstra)

**复现方案**：
```python
class GTGCostNet(nn.Module):
    """学习edge cost"""
    def __init__(self):
        self.edge_encoder = MLP(edge_features, hidden, 1)
    
    def forward(self, edge_feat):
        return self.edge_encoder(edge_feat)  # predicted cost

# 训练：让shortest path under learned cost = GT route
# 推理：A* search with learned cost
```

**实现难度**：中等（需要实现differentiable shortest path）

**输出**：`_sync/wsa/sota/S1_gtg/`

---

### S2: DiffTraj / Cardiff ⬜ 待执行
**论文参考**：
- DiffTraj (KDD 2024): Diffusion for trajectory generation
- Cardiff (ICLR 2024): Diffusion with graph constraints

**核心思路**：
- 在GPS空间做diffusion
- 用graph约束做guidance/projection

**复现方案（简化版）**：
```python
class DiffTrajModel(nn.Module):
    """GPS空间diffusion"""
    def __init__(self):
        self.denoiser = UNet1D(...)  # denoise GPS sequence
        self.cond_enc = ConditionEncoder(...)
    
    def forward(self, x_t, t, cond):
        # x_t: (B, T, 2) noisy GPS
        return self.denoiser(x_t, t, cond)

# 推理后：snap to nearest way (graph projection)
```

**与Way-CASD的关键区别**：
- 在GPS空间而非way空间生成
- 无hierarchy（无region）
- Diffusion vs Flow Matching

**实现难度**：中等

**输出**：`_sync/wsa/sota/S2_difftraj/`

---

## 三、Ablation Studies (我们的变体)

### A1: Way-CASD (no region) ⬜ 待执行
**目的**：证明region hierarchy的价值

**方法**：
- 使用相同的AE + Flow
- 评估时关闭region_constraint
- Flow训练时关闭use_region_seq

**配置**：
```yaml
flow: use_region_seq=False
eval: region_constraint=none
```

**输出**：`_sync/wsa/ablations/A1_no_region/`

---

### A2: Way-CASD (no past_k) ⬜ 待执行
**目的**：证明past context的价值

**方法**：
- 使用past_k=0的AE (W8 or 重训)
- 其他配置相同

**输出**：`_sync/wsa/ablations/A2_no_past/`

---

### A3: Way-CASD (no latent) ⬜ 待执行
**目的**：证明latent compression的价值

**方法**：
- 直接用condition做AR，无latent
- 类似B3但加上region约束

**输出**：`_sync/wsa/ablations/A3_no_latent/`

---

## 四、统一评估协议

所有模型使用**完全相同的评估**：

```bash
python src/evaluation/way_casd_binned_eval.py \
  --model_type {baseline/sota/ours} \
  --ckpt {model_ckpt} \
  --n_routes 200 \
  --min_hops 5 \
  --max_way_len 160 \
  --beam_size 10 \
  --seed 0 \
  --output {output_json}
```

**评估指标**：
- Success Rate (按bin)
- DTW / Fréchet distance
- hit_wall_rate, loop_rate
- Route length ratio

---

## 五、执行计划

### Phase 1: 简单Baselines (1-2天)
```
B1: Shortest Path    → Partner A (0.5天)
B2: RNN AR           → Partner A (1天)
B3: Transformer AR   → Partner A (1天)
```

### Phase 2: SOTA复现 (3-5天)
```
S1: GTG              → Partner B (2-3天)
S2: DiffTraj         → Partner B (2-3天)
```

### Phase 3: Ablations (1天)
```
A1: no region        → 直接eval (0.5天)
A2: no past_k        → 需重训AE (1天)
A3: no latent        → 复用B3+region (0.5天)
```

**总计：5-8天可完成所有comparison实验**

---

## 六、预期结果表格

| Model | Type | [5,10) | [10,20) | [20,30) | [30,40) | [40,60) | [60,+) |
|-------|------|--------|---------|---------|---------|---------|--------|
| Shortest Path | Baseline | - | - | - | - | - | - |
| RNN AR | Baseline | - | - | - | - | - | - |
| Transformer AR | Baseline | - | - | - | - | - | - |
| GTG | SOTA | - | - | - | - | - | - |
| DiffTraj | SOTA | - | - | - | - | - | - |
| Way-CASD (no region) | Ablation | - | - | - | - | - | - |
| Way-CASD (no past) | Ablation | - | - | - | - | - | - |
| **Way-CASD (ours)** | **Ours** | - | - | - | - | **50%** | - |

---

## 七、代码组织

```
src/
  baselines/
    shortest_path.py      # B1
    rnn_ar.py             # B2
    transformer_ar.py     # B3
  sota/
    gtg.py                # S1: GTG复现
    difftraj.py           # S2: DiffTraj复现
  evaluation/
    unified_eval.py       # 统一评估接口
```

---

## Partner执行指南

### B1: Shortest Path (最简单，先做)
```bash
# 1. 实现
# src/baselines/shortest_path.py (见上面的代码)

# 2. 评估
python src/evaluation/baseline_eval.py \
  --method shortest_path \
  --way_graph_npz {path} \
  --way_routes_npz {path} \
  --output _sync/wsa/baselines/B1_shortest_path/results.json
```

### B2: RNN AR
```bash
# 1. 训练
python src/training/train_rnn_ar.py \
  --way_routes_npz {path} \
  --way_graph_npz {path} \
  --way_features_npz {path} \
  --n_epochs 50 \
  --output_dir _sync/wsa/baselines/B2_rnn_ar/

# 2. 评估
python src/evaluation/baseline_eval.py \
  --method rnn_ar \
  --ckpt _sync/wsa/baselines/B2_rnn_ar/ckpt_best.pt \
  ...
```

### S1: GTG
```bash
# 需要先阅读原论文，复现核心算法
# 关键：differentiable shortest path
```

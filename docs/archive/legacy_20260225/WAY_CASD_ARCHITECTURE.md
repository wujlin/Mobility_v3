# Way-CASD: Candidate-Aware Sequential Decoding for Route Generation

## 核心问题

**任务**：给定起点 O、终点 D 及时空条件（小时、星期），生成符合真实人类移动模式的道路级路线序列。

**挑战**：
1. **多源信息融合**：道路网络包含几何（位置、方向、长度）、类别（tier、highway_type）、语义（POI、路面概率）等多源特征，如何有效利用？
2. **约束解码**：生成的路线必须拓扑可行——每一步只能选择当前道路的邻居作为下一跳。
3. **分叉点决策**：在路网分叉点（outdeg ≥ 2），模型需要从多个合法候选中选择正确的下一跳。

---

## 数据表示

### Way-Level Graph

将 OSM 道路网络抽象为有向图 $G = (V, E)$：
- **节点**：每条 way（道路段）
- **边**：way 之间的拓扑连接关系（共享节点）

### Way Features（多源特征）

每条 way $w$ 具有特征向量 $\mathbf{f}_w$：

| 特征类别 | 维度 | 说明 |
|---------|------|------|
| **几何** | 5 | center_yx (2), dir_yx (2), log1p(len_m) (1) |
| **类别** | 2 | tier (embedding), highway_code (embedding) |
| **语义** | 5 | road_prob_major/minor/service, entropy, poi_total |

总计 d_way 维特征，通过 MLP 投影到 d_model=256。

### Route Data

每条路线 $\mathbf{r} = (w_1, w_2, \ldots, w_T)$ 附带条件：
- start_pos / dest_pos：起终点坐标
- hour / dow：出发时间（小时、星期）
- route_city：城市编码

---

## 模型架构

### 整体设计：Two-Stage Autoencoder

```
┌─────────────────────────────────────────────────────────────┐
│                     Way-CASD Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: GT route (w₁, w₂, ..., wₜ) + conditions (O,D,t)     │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   WayEncoder                         │    │
│  │  way_emb → Transformer → mean-pool → z_enc (L×d)    │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│                    Latent z_enc                              │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   WayDecoder                         │    │
│  │  Constrained AR: at each step t                      │    │
│  │    1. Get candidates C_t = successors(w_{t-1})       │    │
│  │    2. Query z_enc with candidate-aware attention     │    │
│  │    3. Score each candidate → softmax → select        │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  Output: Generated route (ŵ₁, ŵ₂, ..., ŵₜ)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Stage 1: WayEncoder

将输入路线压缩为潜在表示 $\mathbf{z}_{\text{enc}} \in \mathbb{R}^{L \times d}$：

```python
way_emb = WayEmbedder(way_ids)           # (T, d_model)
z_seq = Transformer(way_emb)              # (T, d_model)
z_enc = LearnableLatent(n_latent=64)      # (L, d_model)
z_enc = CrossAttn(z_enc, z_seq)           # (L, d_model)
```

**设计选择**：
- 使用 L=64 个可学习的 latent tokens 而非直接用序列均值
- 这允许模型学习如何从变长序列中提取固定维度的信息

### Stage 2: WayDecoder

核心创新在解码器。每一步 $t$ 的决策流程：

#### 2.1 候选获取（拓扑约束）

```python
candidates = graph.successors(current_way)  # 拓扑合法的下一跳
```

#### 2.2 候选感知 Cross-Attention（核心创新）

**问题**：传统设计中，context $\mathbf{c}_t$ 对所有候选相同：

```python
# 传统设计（候选无关）
query = cur_emb + step_emb + dest_proj + past_ctx
ctx = CrossAttn(query, z_enc)    # (T, d) → 对所有候选相同
ctx_expanded = ctx.expand(T, C, d)  # z_enc 只作为"全局偏置"
```

这导致 $\mathbf{z}_{\text{enc}}$ 中的多源信息**无法直接参与候选排序**。

**解决方案**：让每个候选独立查询 $\mathbf{z}_{\text{enc}}$：

```python
# 候选感知设计
base_query = cur_emb + step_emb + dest_proj + past_ctx  # (T, d)
cand_query = base_query[:, None, :] + cand_proj(cand_emb)  # (T, C, d)

# 每个候选独立 cross-attention
for c in range(C):
    ctx[t, c] = CrossAttn(cand_query[t, c], z_enc)
```

**效果**：不同候选可以从 $\mathbf{z}_{\text{enc}}$ 提取**与自身相关**的不同信息。

#### 2.3 打分与选择

```python
# 候选对比特征：让 scorer 看到“该候选相对其他候选的区别”
mean_cand = masked_mean(cand_h, mask=cand_mask)             # (T, d)
diff_from_mean = cand_h - mean_cand[:, None, :]             # (T, C, d)

x = concat([ctx_c, cur_h, cand_h, diff_from_mean, dest_dist])  # (T, C, 4d+1)
logits = Scorer(x)  # MLP → (T, C)
next_way = candidates[argmax(logits)]
```

### 辅助模块

| 模块 | 作用 |
|------|------|
| **PastContextEncoder** | Transformer 编码过去 K 步历史 |
| **ConditionEncoder** | 融合 O/D 坐标 + 时间 + 城市 |
| **dest_dist** | 候选到终点的欧氏距离特征 |

---

## 核心 Insight

### Insight 1: 多源数据的价值在于"能被决策利用"

**观察**：
- 传统架构：$\mathbf{z}_{\text{enc}}$ 编码了多源信息，但在解码时 `.expand()` 到所有候选
- 结果：$\mathbf{z}_{\text{enc}}$ 只能作为"全局偏置"，无法区分候选

**数据验证**：

| 配置 | 到达率 | Δ |
|------|--------|---|
| candq=0（候选无关） | 58.25% | baseline |
| candq=1（候选感知） | **82.25%** | **+24.0pp** |

> 口径注释（避免误读）：以上数字来自 `z_enc informativeness` 诊断（GT→Encoder→latent→Decoder 的 *oracle/reconstruction* 上界），对应文件：  
> - candq=0：`_sync/wsa/icml2026_routegen/WAYCASD_AB_candquery_strict_sem5_seed0_e100/WAYCASD_AB_candq0_pastctx_k8_strict_sem5_seed0_e100/W8_diag/zenc_info_n200.json`  
> - candq=1：`_sync/wsa/icml2026_routegen/WAYCASD_AB_candquery_strict_sem5_seed0_e100/WAYCASD_AB_candq1_pastctx_k8_strict_sem5_seed0_e100/W8_diag/zenc_info_n200.json`  
> 且该实验 **未采用 min_hops=5 的短路线过滤**；若论文主口径采用 min_hops=5（见 `docs/WAYCASD_EXPERIMENT_LOG.md` 第 0.0 节），需要在同口径下重跑 ablation 才能直接放主表。

| z_enc 条件 | candq=0 | candq=1 |
|------------|---------|---------|
| true | 58.25% | 82.25% |
| shuffle | 16.25% | 11.25% |
| zero | 18.0% | 13.5% |

**结论**：
- candq=1 的 true vs shuffle 差异更大（71pp vs 42pp），说明多源信息被更有效利用
- shuffle/zero 性能更低（~12%），证明提升不是"瞎猜"

### Insight 2: 剩余失败的瓶颈在候选嵌入区分度

**Attention 诊断**（candq=1 的 71 例失败）：

| 分组 | n | 占比 | 特征 |
|------|---|------|------|
| cos(pred_attn, gt_attn) ≥ 0.95 | 38 | 53.5% | attention 几乎相同，但仍排错 |
| cos < 0.95 | 33 | 46.5% | attention 差异大 |

高相似组分析：
- `cand_h_diff = 1.85`（两个候选嵌入几乎相同）
- `logit_gap = 0.35`（scorer 面对相似输入做错排序）

**结论**：attention 已正常工作，瓶颈在**候选嵌入本身的区分度不足**。

### Insight 3: 跨城市泛化

| 城市 | Seed 0 | Seed 1 | Seed 2 | Mean |
|------|--------|--------|--------|------|
| Detroit | 74.5% | 64.5% | 71.5% | 70.2% |
| Columbus | 84.0% | 74.5% | 79.0% | 79.2% |

- Detroit 始终难于 Columbus（~9pp gap）
- 可能原因：网络更复杂 / 语义特征质量差异

---

## 实验配置

```yaml
# 数据
dataset: WorldTrace Rust Belt (Detroit + Columbus)
n_routes: 5392 (train=4853, val=539)
max_way_len: 160
way_semantic: 5 channels (road_prob_major/minor/service, entropy, poi_total)

# 模型
d_model: 256
n_latent: 64
pastctx_k: 8
decoder_use_cand_query: true
decoder_use_dest_dist: true
decoder_use_past_context: true

# 训练
n_epochs: 100
optimizer: AdamW
seed: 0, 1, 2
```

---

## 关键代码路径

| 组件 | 文件 | 关键行 |
|------|------|--------|
| WayEncoder | `src/models/way_casd/way_encoder.py` | WayEncoder 类 |
| WayDecoder | `src/models/way_casd/way_decoder.py` | `_compute_context()` (L320-450) |
| 候选感知 Cross-Attn | `way_decoder.py:410-435` | `cand_query_proj` 实现 |
| 训练入口 | `src/training/train_way_casd_autoencoder.py` | `--decoder_use_cand_query` |
| 诊断脚本 | `src/evaluation/way_casd_oracle_step_diagnose.py` | attention dump |

---

## 下一步方向

1. **增强候选嵌入对比性**：在 scorer 输入中增加候选间对比特征
2. **Detroit 专项诊断**：分析城市间性能差异的根因（网络结构 vs 特征质量）
3. **Beam Search 评估**：验证 beam_size > 1 时的性能上限

# Way-CASD 实验 Checklist

> 目标：缩小 Oracle (85.3%) → Flow (50%) 的 35pp gap
> 更新日期：2026-02-05

---

## 当前最佳配置

```yaml
AE: W9 (past_k=16, cand_query=True)
Flow: W11 (RegionSeq xattn, n_layers=6)
Decode: beam=10, soft P=2.0 K=4, maxcand=0
Region: AR constraint, relaxed, dest_region fallback
```

**当前性能**：
| Metric | [60,+) Beam | Overall |
|--------|------------|---------|
| E1 Flow v3 | 50.0% | ~52% |
| Oracle | 85.3% | ~92% |
| **Gap** | **35.3pp** | **~40pp** |

---

## 核心问题诊断

**关键发现**：短路线gap反而更大
- [5,10): Oracle=100%, Flow=47.9%, Gap=52.1pp
- [60,+): Oracle=85.3%, Flow=50.0%, Gap=35.3pp

**结论**：问题不是"长路线更难"，而是 **Flow latent与GT latent存在系统性分布偏移**，Decoder对这种偏移不鲁棒。

---

## Phase E: Flow-Decoder Gap 攻坚

### E1: Latent分布诊断 ⬜ 待执行
**目标**：量化z_flow与z_gt的分布差异，验证假设

**方法**：
1. 对验证集的所有routes，分别获取：
   - z_gt = AE.encode(route)
   - z_flow = Flow.sample(condition)
2. 计算统计指标：
   - Per-sample MSE: ||z_flow - z_gt||²
   - Per-token cosine similarity
   - 按route_len分bin统计
3. 可视化：t-SNE/PCA对比z_gt和z_flow分布

**输出**：`_sync/wsa/pi_verify/E1_latent_diagnosis/`
- `latent_stats.json`: MSE/cosine统计
- `latent_tsne.png`: 可视化

**预期**：
- 如果z_flow和z_gt分布差异大 → 需要改进Flow
- 如果z_flow和z_gt分布相近但decoder仍失败 → 需要Joint FT

---

### E2: Flow-Decoder Joint Fine-tuning ⬜ 待执行
**目标**：让Decoder适应Flow生成的latent分布

**方法**：
```python
# 核心思路：用Flow latent fine-tune decoder
for batch in dataloader:
    z_flow = flow.sample(route_cond)  # 生成latent
    # Option A: Teacher forcing (GT next-token, 但用z_flow)
    loss = decoder.compute_loss(z_enc=z_flow, way_seq=gt_seq)
    # Option B: RL (让decoder在z_flow下学习到达dest)
    loss = decoder_rl_loss(z_flow, batch)
```

**实验配置**：
- 基于W9 AE + W11 Flow
- Fine-tune decoder 10-20 epochs
- 使用较小lr (1e-5) 防止灾难性遗忘

**输出**：`_sync/wsa/pi_verify/E2_joint_finetune/`
- `ckpt_joint.pt`: fine-tuned decoder
- `binned_joint.json`: 评估结果

**预期收益**：+10-15pp on [60,+)

---

### E3: CFG推理验证 ⬜ 可选
**目标**：快速验证condition利用是否充分（成本低）

**方法**：
```python
# 修改 LatentFlowMatching.sample()
def sample_cfg(self, route_cond, cfg_scale=2.0):
    # 训练时需要10%概率drop condition
    v_uncond = self._v(z, t, empty_cond)
    v_cond = self._v(z, t, route_cond)
    v = v_uncond + cfg_scale * (v_cond - v_uncond)
```

**注意**：需要先重训Flow with condition dropout

**预期收益**：+3-5pp（如果condition利用不充分）

---

## Phase D: 已完成实验总结

### D1: Hit-Wall Sweep ✅
| Config | [60,+) Beam | 结论 |
|--------|------------|------|
| alpha=0.0 (baseline) | 44.1% | baseline |
| maxcand=0 | **47.1%** | **+3pp, 推荐长程** |
| maxcand=5 | 41.2% | -2.9pp |
| alpha=0.1/0.2/0.3 | 35.3% | 显著下降，不推荐 |

**结论**：`decode_max_candidates=0` 对长程路线有正向作用

### D2: RL Reward Redesign ✅
- 新增penalties: turn=0.05, hit_wall=1.0, wall_prox=0.2, margin=20
- [60,+) beam = 44.1% (与baseline持平)
- **结论**：简单reward penalty无效，问题在latent质量

### D3: City Audit ✅
| City | n_routes | route_len p50 | route_len p90 |
|------|----------|---------------|---------------|
| Detroit | 1,394 | 35 | 60 |
| Columbus | 3,628 | 22 | 50 |

**结论**：Columbus路线显著更短，解释了per-city性能差异

---

## 放弃的实验方向

| 实验 | 原因 |
|------|------|
| Latent Regularization | 治标。AE本身没问题（Oracle 85%），问题在Flow |
| route_len_bin condition | 信息泄露。真实场景不知道目标长度 |
| Curriculum Learning (短→长) | 方向错误。短路线gap反而更大，非复杂度问题 |
| Multi-Scale Flow | 成本高。region_seq xattn已提供coarse信息 |

---

## 执行优先级

```
P0 (立即执行):
  E1: Latent分布诊断 → 验证假设，指导后续方向
  E2: Flow-Decoder Joint FT → 核心治本实验

P1 (E2完成后):
  根据E1/E2结果决定是否需要改进Flow本身

P2 (可选):
  E3: CFG推理 → 如果condition利用不充分
```

---

## Partner执行指南

### E1 执行步骤
```bash
# 1. 创建诊断脚本
python src/evaluation/latent_diagnosis.py \
  --ae_ckpt W9 \
  --flow_ckpt W11 \
  --n_routes 400 \
  --output_dir _sync/wsa/pi_verify/E1_latent_diagnosis/

# 2. 输出文件
# - latent_stats.json: 统计指标
# - latent_tsne.png: 可视化
```

### E2 执行步骤
```bash
# 1. Joint fine-tune
python src/training/train_way_casd_decoder_joint.py \
  --ae_ckpt W9 \
  --flow_ckpt W11 \
  --mode teacher_forcing \
  --lr 1e-5 \
  --n_epochs 20 \
  --output_dir _sync/wsa/pi_verify/E2_joint_finetune/

# 2. 评估
python src/evaluation/way_casd_binned_eval.py \
  --ae_ckpt E2_joint_finetune/ckpt_best.pt \
  --flow_ckpt W11 \
  --region_constraint ar \
  ...
```

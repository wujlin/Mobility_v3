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
- `latent_pca.png`: 可视化（PCA，避免额外依赖）

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

### D4: [40,60) hit_wall 深挖（空间模式）⬜ 待执行
**目标**：回答 hit_wall 是否集中在特定空间区域/高出度交叉口（outdegree proxy）

**方法**：
1) 先用 `way_casd_binned_eval.py` 生成 `per_route.json`（务必 `--dump_way_seqs`）
2) 再运行空间审计脚本 `tools/waycasd_analyze_hit_wall_spatial.py`

**输出**：`_sync/wsa/pi_verify/D4_hit_wall_spatial/`
- `hit_wall_spatial_audit.json`
- `hit_wall_spatial.png`

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
# 0) 约定：提前 export 这些变量（示例）
# RAW_ROOT=...; WAY_ROUTES_NPZ=...; WAY_GRAPH_NPZ=...; WAY_FEATS_NPZ=...
# AE_CKPT=...; FLOW_CKPT=...; WAY_REGIONS_NPZ=...

# 1) E1: latent mismatch 诊断（Flow 若 use_region_seq，则必须传 --way_regions_npz）
PYTHONUNBUFFERED=1 python -u -m src.evaluation.latent_diagnosis \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$AE_CKPT" \
  --flow_ckpt "$FLOW_CKPT" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --out_dir "_sync/wsa/pi_verify/E1_latent_diagnosis_s0" \
  --n_routes 200 --min_hops 5 --max_way_len 160 \
  --batch_size 64 \
  --device cuda --seed 0 \
  |& tee "_sync/wsa/pi_verify/E1_latent_diagnosis_s0/run.log"

# 2) 输出文件
# - latent_stats.json / latent_pairs.npz / latent_pca.png
```

### E2 执行步骤
```bash
# 1) Joint fine-tune（只训 decoder.*，用 z_flow 做 latent tokens，teacher forcing CE）
PYTHONUNBUFFERED=1 python -u -m src.training.train_way_casd_decoder_joint \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$AE_CKPT" \
  --flow_ckpt "$FLOW_CKPT" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --out_dir "_sync/wsa/pi_verify/E2_joint_finetune_s0" \
  --min_hops 5 --max_way_len 160 \
  --batch_size 64 --num_workers 16 \
  --n_epochs 20 --lr 1e-5 --weight_decay 0.0 --val_ratio 0.1 \
  --device cuda --seed 0 \
  |& tee "_sync/wsa/pi_verify/E2_joint_finetune_s0/run_train.log"

# 2) 评估：用 binned eval 复现论文口径（Flow end-to-end）
PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "_sync/wsa/pi_verify/E2_joint_finetune_s0/ckpt_best.pt" \
  --latent_source flow --flow_ckpt "$FLOW_CKPT" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --region_constraint ar --region_ar_ckpt "$REGION_AR_CKPT" \
  --region_constraint_mode relaxed --region_constraint_fallback dest_region \
  --n_routes 200 --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --decode_candidate_policy first --decode_max_candidates 0 \
  --beam_size 10 \
  --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
  --out_json "_sync/wsa/pi_verify/E2_joint_finetune_s0/binned_eval_flow_n200pc.json" \
  |& tee "_sync/wsa/pi_verify/E2_joint_finetune_s0/run_eval.log"
```

### D4 执行步骤（[40,60) hit_wall 空间审计）
```bash
# 1) 生成 per-route dump（建议用当前 best Flow 配置；要空间审计必须 --dump_way_seqs）
PYTHONUNBUFFERED=1 python -u -m src.evaluation.way_casd_binned_eval \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$AE_CKPT" \
  --latent_source flow --flow_ckpt "$FLOW_CKPT" \
  --way_regions_npz "$WAY_REGIONS_NPZ" \
  --region_constraint ar --region_ar_ckpt "$REGION_AR_CKPT" \
  --region_constraint_mode relaxed --region_constraint_fallback dest_region \
  --n_routes 200 --min_hops 5 --max_way_len 160 --max_decode_len 160 \
  --decode_candidate_policy first --decode_max_candidates 0 \
  --beam_size 10 \
  --anti_loop_penalty 2.0 --anti_loop_penalty_k 4 \
  --out_json "_sync/wsa/pi_verify/D4_hit_wall_spatial_s0/binned_eval_flow_n200pc.json" \
  --out_per_route_json "_sync/wsa/pi_verify/D4_hit_wall_spatial_s0/per_route_flow_n200pc.json" \
  --dump_way_seqs \
  |& tee "_sync/wsa/pi_verify/D4_hit_wall_spatial_s0/run_eval.log"

# 2) 做空间审计（默认分析 [40,60) 桶）
PYTHONUNBUFFERED=1 python -u tools/waycasd_analyze_hit_wall_spatial.py \
  --per_route_json "_sync/wsa/pi_verify/D4_hit_wall_spatial_s0/per_route_flow_n200pc.json" \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --out_dir "_sync/wsa/pi_verify/D4_hit_wall_spatial_s0/audit" \
  --key beam --hops_bin "[40,60)" \
  |& tee "_sync/wsa/pi_verify/D4_hit_wall_spatial_s0/audit/run_audit.log"
```

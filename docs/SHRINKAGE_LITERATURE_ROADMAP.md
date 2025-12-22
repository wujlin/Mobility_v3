# Shrinkage / Macro–Micro Trade-off：文献驱动的下一步路线图（2023–2024）

> 目的：把我们当前遇到的 **macro–micro 权衡**（宏观真实性 vs 微观误差）放回到 2023–2024 顶会/期刊的共同语境中，给出 **可执行、可证伪、成本可控** 的下一步路线。  
> 原则：KISS；不引入无法归因的变量；先做低成本验证，再做高成本结构升级。

---

## 0) 当前事实（我们到底卡在哪里）

我们已经把 v1.0 的“走不动（shrinkage）”从猜测变成事实链，并且通过 v1.2（diff\_loss 位移加权）证明它是可控的：

- **现象 A（已证实）**：无约束/弱约束下，diffusion 会向均值回归，表现为 **速度、Rog、MSD10 偏低**（macro shrinkage）。
- **现象 B（已证实）**：继续训练会出现 **micro 变好但 macro 变差** 的 safe-play（局部最优陷阱）。
- **现象 C（已证实）**：v1.2 通过对 diff\_loss 做位移加权（允许 `w>1`）可以显著抬升 macro，但会带来 micro 代价，并可能出现轻微过冲（ratio > 1）。

结论：我们已经从“能不能走得动”进入“如何在 Pareto 前沿上选点/更稳地训练到更优点”。

---

## 1) 文献中如何称呼这个问题

文献常用的病理术语与我们的对齐如下：

- **Mean-Reversion / Mode Collapse**：模型倾向预测均值（safe-play），多样性不足或样本偏保守。
- **Displacement Underestimation**：净位移系统性偏小（MSD/Rog 下降）。
- **Directional Persistence Deficit**：方向持久性不足，表现为 MSD 曲线偏低（类似随机游走抵消）。

---

## 2) 六条改进方向：我们该做什么、不该做什么

> 下文每个方向都给出：**收益**、**成本**、**风险**、**是否踩过坑**、**最小验证**（避免盲目扫参）。

### 2.1 v-Prediction 替代 ε-Prediction（P1，低成本，优先）

**核心想法**  
当前我们用 ε-pred（预测噪声）。v-pred 预测的是噪声与数据的线性组合，通常对低频结构更友好：

- ε-pred（当前）：`loss = MSE(ε̂, ε)`
- v-pred（建议）：`v = α_t * ε - σ_t * x0`，`loss = MSE(v̂, v)`

其中 `x_t = α_t x0 + σ_t ε`，可由 `v̂` 恢复：

- `x0_pred = α_t x_t - σ_t v̂`
- `ε_pred = σ_t x_t + α_t v̂`

**预期收益**  
- 训练更稳定（尤其在高噪声 t 段），更可能保留低频（位移尺度）结构；
- 对 shrinkage / safe-play 有机会带来“更少的代价”（降低 macro–micro 冲突）。

**成本**  
- 代码改动小：只需改训练 target 与采样时的输出解释；模型结构不变。

**风险**  
- 未必在我们的数据/网络上必然改善（需要最小对照证伪）；但风险低。

**我们是否踩过坑**  
- 未做过（属于“新变量”，但不引入架构变化）。

**最小验证（必须做）**  
保持一切不变，只切换 `pred_type=eps → v`：
- `train: e20, max_batches=200`
- `eval: val/test K=10, max_batches=200, diff_steps=100`
- 成功判据：macro ratio 回升或同等 macro 下 micro 改善（至少不更差）。

---

### 2.2 CFG（Classifier-Free Guidance）+ Destination Guidance（P1，中成本）

**核心想法**  
KnownDestination 任务下，destination 是强条件。CFG 在采样时显式放大条件差分：

`pred = pred_uncond + s * (pred_cond - pred_uncond)`

**预期收益**  
- 推理期可控“拉向目的地”，能更原理化地调节 macro（相比 vel\_scale 更可解释）；
- 有机会改善“走不到目的地”的样本（FDE / EPE）。

**成本**  
- 需要训练期做 **condition dropout**（得到 uncond 分支能力）；
- 推理期每步需要两次前向（成本约 ×2）。

**风险**  
- 若 dropout 设计不当，uncond 分支不稳，guidance 反而引入偏差或抖动；
- 与 nav_field 的 mean-field 作用可能耦合（需要严格对照）。

**我们是否踩过坑**  
- 我们已经证伪过“调 temperature 拉宏观”，CFG 不等于 temperature（不同机制），仍值得做。

**最小验证**  
先做最简 CFG（只 dropout destination 两维），在 residual physics 上：
- 训练：`cond_drop_prob=0.1`（仅目的地维度）
- 推理：`guidance_scale ∈ {0, 1, 2}` 只做 3 点对照
- 成功判据：macro 不过冲的情况下 FDE_best 明显下降，且不产生高频抖动。

**当前仓库状态（已落地）**  
- 训练支持：`src/training/train_diffusion.py` 已增加 `--cfg_drop_dest_prob` / `--cfg_uncond_dest_mode`  
- 评估支持：`src/training/evaluate.py` 已增加 `--cfg_scale` / `--cfg_uncond_dest_mode`  
- 一键脚本：`scripts/phase_b_cfg_destination_guidance.sh`

---

### 2.3 OGD 思想：更好的“起点分布”与更少采样步数（P2，偏效率）

**定位**  
OGD/加速主要解决 **采样效率**。我们的当前瓶颈是 validity vs micro 的权衡，优先级低于 v-pred/CFG。

**最小验证**  
在 residual 框架下可直接做：
- 用 `diff_steps=100` 做质量口径；
- 额外报告 `diff_steps=30/50` 作为工程加速（仅作附录/落地讨论）。

> 注意：我们已经观察过 ds50 会产生分布漂移，不能用来“宣布效果更好”。

---

### 2.4 LED：Learnable Initializer（P2，结构升级）

**核心想法**  
我们目前的 prior 是 frozen deterministic baseline（fixed initializer）。LED 把 initializer 变为可学习并与 diffusion 联训。

**风险/成本**  
- 会引入“prior 也在变”的变量，短期难以归因；
- 适合在 v-pred/CFG 确认无力后再上。

---

### 2.5 Distributional Diffusion（P3，高成本）

**核心想法**  
不在坐标空间生成，而在分布参数空间（μ, Σ）上扩散，直接建模不确定性。

**定位**  
更适合 v2，当前不建议做。

---

### 2.6 Flow Matching（P3，高成本但潜力大）

**核心想法**  
直接学习从噪声到数据的速度场，路径更“直”，对低频结构/方向持久性更友好。

**定位**  
v2 路线（需要较大重构），适合作为 future work 或下一篇文章。

---

## 3) 是否需要引入多模态数据（遥感/POI/路网）

结论（当前阶段）：**不需要作为主线修复**。

原因（第一性原理）：
- 我们当前的主要误差来源是 **目标函数/分布偏置**（safe-play、过冲、macro–micro tradeoff），不是信息不足；
- 引入 POI/遥感会带来稀疏多模态与工程复杂度，反而可能加剧均值坍缩（预测重心）；
- 路网约束（map-matching/graph diffusion）是任务边界升级（v2），适合后续。

建议写法：
- 在论文叙事中把我们的模型定位为 Digital Twin 的“生成式轨迹模块”；
- 多模态（路网/POI/遥感）作为未来工作：提升可行性与语义一致性。

---

## 4) nav_field 注入我们做到了哪里（避免重复踩坑）

已完成：
- PhysicsConditionDiffusion：`nav_patch → CNNEncoder → nav_emb`，与 `cond` concat 注入（主线）。
- ablation：`nav_patch_channel2`（speed/count/zeros）、`nav_emb_scale`、`NavGate(obscond)`。

已止损（不要再反复提）：
- 温度（temperature）调参想象：只会引入 jitter；
- 高噪声 t 段强行压 macro-loss：会走捷径（高频抖动满足宏观）。

---

## 5) 更新后的 Action Items（建议）

P1（优先，低成本、可归因）：
1) v-prediction（eps vs v）最小对照
2) CFG（destination dropout + guidance\_scale 三点）

P2（在 P1 结论明确后再做）：
3) 采样加速（DPM-Solver/DDIM/OGD 思路）作为效率附录
4) learnable initializer（LED 风格）

P3（未来工作）：
5) distributional diffusion / flow matching

---

## 6) 可复现实验命令（v-pred 最小对照，建议立刻跑）

> 目标：只切换 `pred_type`（eps vs v），其余全部固定，验证是否能**在不牺牲太多 micro 的情况下**改善 macro（shrinkage / safe-play）。
>
> 评估脚本已支持 `--pred_type auto`：会从 checkpoint 的 `config.pred_type` 自动对齐，避免“权重是 v，但按 eps 解释”的错误评估。

### 6.1 Residual Physics（主线模型，推荐）

```bash
export DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
export NAV=data/processed_dt30/nav_field.npz
export PRIOR=data/experiments/baseline_b_dt30/last.pt

# ---- Train (Fast Check) ----
# 口径：e20 + max_batches=200（只为快速证伪/证实趋势）
python -m src.training.train_diffusion \
  --model_type physics \
  --data_path $DATA --nav_file $NAV --split train \
  --prior_checkpoint $PRIOR \
  --exp_name phys_residual_predEps_e20_mb200_s0 \
  --hidden_dim 128 --batch_size 2048 --lr 1e-3 --epochs 20 \
  --max_batches 200 --num_workers 16 --seed 0 \
  --pred_type eps

python -m src.training.train_diffusion \
  --model_type physics \
  --data_path $DATA --nav_file $NAV --split train \
  --prior_checkpoint $PRIOR \
  --exp_name phys_residual_predV_e20_mb200_s0 \
  --hidden_dim 128 --batch_size 2048 --lr 1e-3 --epochs 20 \
  --max_batches 200 --num_workers 16 --seed 0 \
  --pred_type v

# ---- Eval (Precision) ----
# 口径：K=10, B=200, diff_steps=100（严谨口径；ds50 只用于快筛，不用于“宣布更好”）
python -m src.training.evaluate \
  --exp_name phys_residual_predEps_val_k10_mb200 \
  --model_type physics \
  --data_path $DATA --nav_file $NAV \
  --checkpoint data/experiments/phys_residual_predEps_e20_mb200_s0/last.pt \
  --prior_checkpoint $PRIOR \
  --split val --batch_size 256 --num_workers 0 --max_batches 200 \
  --num_samples_per_condition 10 --diff_steps 100 --save_samples 0 --seed 0 \
  --pred_type auto

python -m src.training.evaluate \
  --exp_name phys_residual_predV_val_k10_mb200 \
  --model_type physics \
  --data_path $DATA --nav_file $NAV \
  --checkpoint data/experiments/phys_residual_predV_e20_mb200_s0/last.pt \
  --prior_checkpoint $PRIOR \
  --split val --batch_size 256 --num_workers 0 --max_batches 200 \
  --num_samples_per_condition 10 --diff_steps 100 --save_samples 0 --seed 0 \
  --pred_type auto
```

### 6.2 通过/失败判据（避免扫参）

- **通过（值得推进）**：`RoG/MSD10` ratio 回升或在同等水平下 `ADE_best/FDE_best` 更好（macro–micro 更接近 Pareto 前沿）。
- **失败（立刻止损）**：`RoG/MSD10` 无明显改善且 `ADE_best/FDE_best` 变差（说明 v-pred 在当前架构/数据上收益不足）。

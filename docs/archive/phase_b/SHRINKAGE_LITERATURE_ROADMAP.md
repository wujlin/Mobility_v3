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

### 2.6 Rectified Flow / Flow Matching（P2，实验性 high-risk）

**核心想法**  
直接学习从噪声到数据的速度场，路径更“直”，对低频结构/方向持久性更友好。

**定位**  
不是当前 diffusion 主线的“必做项”，但在 PI 批准下可做一个 **24h time-box 的 pilot**：  
用 **Physics Residual RF** 做 A/B（20-step ODE vs 100-step diffusion），验证是否能更好地保留低频位移结构并显著加速推理。  
细节与止损标准见：`docs/archive/phase_b/RF_PILOT.md`。

**阶段性结论（2025-12 pilot）**  
在 `Val, K=10, max_batches=200` 的可比口径下，RF@20 steps（Euler, no-CFG）在 micro 与 macro 两侧均落后于 diffusion+CFG（详见 `docs/archive/phase_b/RF_PILOT.md`），因此本轮按 time-box 原则止损暂停，避免无归因扫参。

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
5) Rectified Flow pilot（Physics Residual，24h time-box）

P3（未来工作）：
6) distributional diffusion（更系统的分布建模；v2 方向）

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

---

## 7) 结合我们当前实证：哪些方向“有效”、哪些“先别碰”

> 这节是为了 **减少重复踩坑**：把“文献可能有效”收敛为“我们实证上确实有效/暂时无效/不该在 v1 阶段做”的结论。

### 7.1 我们已经验证为有效（P0）

1) **Residual（Prior + Residual）是必要架构**  
   - 事实：纯 diffusion/physics 在 dt30 上出现系统性 shrinkage；Residual 直接把尺度锚定到 prior，避免“走不动”。  
   - 注意：Residual 的上限受 prior 质量影响（已被你们的 prior 位移加权实验确认）。

2) **CFG（目的地引导）是有效的推理期旋钮**  
   - 事实：cfg=2/3 在同一评估口径下呈现稳定的 micro–macro trade-off（并非偶然）。  
   - 重要边界：CFG 只能放大“朝目的地的平均梯度场”，**并不会凭空产生低频 detour/拓扑选择**；在 trip-level 诊断里容易表现为 Destination Gravity（直冲终点）+ 抖动。  
     - 结论口径：CFG 可作为 Phase B/窗口级的对照旋钮与可视化素材，但**不应再作为 trip-level 主线去赌“调参能绕路”**（证据链见 `docs/archive/phase_b/PHASE_B_CFG_VISUALIZATION.md`、`docs/PHASE_C_RESULTS.md`）。  
   - 口径建议：不再扫 cfg 网格；固定两点：  
     - 主表：cfg=2（micro-optimal within macro validity gate）  
     - 附图：cfg=3（macro-validity-optimal，展示可调性）

3) **Prior 的位移加权（clip, w>1）能显著抬升宏观尺度**  
   - 事实：`disp_weight=clip(0.5,5)` 的 prior 在 test 上能把 speed/Rog/MSD10 从 <1 拉到 ≈1（甚至略过冲）。  
   - 结论：v1.x 阶段的性价比最高做法是“先修 prior，再做 residual”。

### 7.2 我们已验证收益有限/不稳定（P1，但谨慎）

1) **v-prediction（eps→v）**  
   - 状态：我们看到它更偏向“推宏观/牺牲微观”的方向（不保证双赢）。  
   - 建议：作为低成本对照保留，但不要作为 Phase B 窗口级主线投入大量 sweep。

2) **训练期 macro-loss（Rog/EPE/multi-point）**  
   - 状态：无门控/无权重时容易出现 jitter 捷径；门控后改善有限。  
   - 建议：除非有明确的 SNR 权重 + 多点约束组合（并且能在 fast check 中证伪/证实），否则优先级低于 CFG/修 prior。

### 7.3 v1 阶段不建议主线投入（P2/P3）

1) **OGD/采样加速（DPM-Solver/DDIM/减少 diff_steps）**  
   - 定位：效率优化，不是质量突破；ds50 已观察到分布漂移风险（只能用于 fast check，不能用于“宣布更好”）。

2) **LED learnable initializer / Distributional diffusion / Flow matching**  
   - 定位：结构级升级（变量多、归因难），适合作为 v2 或 future work。

---

## 8) CFG 是否有泛化风险？（有，但可控）

### 8.1 风险是什么（第一性原理）

CFG 本质是在 **推理期改变采样分布**：不同 cfg 会生成不同的轨迹族。因此泛化风险主要来自：

- **数据分布变化**：OD 距离分布、出行时段、区域交通走廊变化时，“固定 cfg”可能过激/过保守；
- **条件强弱变化**：某些 OD 本身多模态很强（路口/高速出入口），cfg 过大容易把 micro 拉偏；
- **与 nav_field 的耦合**：physics 条件本身是 mean-field prior，cfg 增强会放大“指向目的地”的趋势，可能导致过冲。

### 8.2 如何把风险变成“可汇报、可复现”的协议

1) **只在 val 上选 cfg，test 上固定**  
   - 我们的推荐两点（cfg=2/3）就是这种协议：cfg=2 主表，cfg=3 作为“可调旋钮”附图。

2) **用“宏观有效性 gate”选 cfg（避免为 ADE 过拟合）**  
   - 规则：在 val 上选“满足 macro ratio 在区间内的最小 cfg”（例如 `Rog_R∈[0.95,1.05]` 且 `MSD10_R∈[0.95,1.05]`）。  
   - 好处：把选择从“扫参”变成“满足物理有效性约束的最小干预”。

3) **按 OD 距离分桶检查（短/中/长位移）**  
   - 目的：确认 cfg=2/3 是否只对某一类 OD 有效。  
   - 若发现偏差：再上 “Adaptive CFG（按 ||d−o|| 分桶选择 cfg）”，比继续全局 sweep 更可泛化。

4) **CFG schedule（推理期随 t 递增）作为下一步突破 micro–macro 张力的低成本方案**  
   - 直觉：高噪声段强 guidance 更伤 micro；后半链路增强更稳。  
   - 这是“推前沿”而不是“扫参数”的关键方向（且不需要重训）。
  --split val --batch_size 256 --num_workers 0 --max_batches 200 \
  --num_samples_per_condition 10 --diff_steps 100 --save_samples 0 --seed 0 \
  --pred_type auto
```

### 6.2 通过/失败判据（避免扫参）

- **通过（值得推进）**：`RoG/MSD10` ratio 回升或在同等水平下 `ADE_best/FDE_best` 更好（macro–micro 更接近 Pareto 前沿）。
- **失败（立刻止损）**：`RoG/MSD10` 无明显改善且 `ADE_best/FDE_best` 变差（说明 v-pred 在当前架构/数据上收益不足）。

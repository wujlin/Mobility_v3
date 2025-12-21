# 专家咨询材料（Phase B / dt=30s）— 当前瓶颈与待解问题（v1.1 之后）

> 目的：把我们已经做过的排雷、事实证据、以及当前真正卡住的问题压缩成一页“可讨论材料”，用于向专家/教授请教。  
> 版本：v1.1 / Phase B（dt-fixed=30s，strict train-only 数据产物）

---

## 0) 一句话结论（当前最硬的事实）

Phase B 的核心问题已经从“跑不通/泄漏”转为“机制性偏差”：  
**Residual decomposition（prior + residual）能结构性修复 data-only diffusion 的宏观收缩，但 physics residual 仍略保守（MSD10/Rog 偏低），我们怀疑主要来自 nav\_field 的 mean-field tether（注入方式过于 naive），而不是 temperature、简单宏观损失或单一通道（speed/count）的锅。**

---

## 1) 严格边界（避免争议点）

- **任务**：KnownDestination（推理时终点 `d` 是合法输入，不是泄漏）。
- **输入/输出**：窗口级预测（`obs_len=8`, `pred_len=12`），输出 future `vel`（**step displacement**）。
- **论文版数据语义**：dt-fixed=30s（Phase B），避免 MSD/速度场语义不明。
- **无泄漏合同**：`data_stats.json` 与 `nav_field.npz` **只用 train split** 估计，并记录 `trajectory_ids_sha256` 等来源信息。

参考：`docs/TASK_DEFINITION.md`

---

## 2) 已完成的排雷验证（我们确认“不是工程错误”）

1) **Split 无重叠**：train/val/test trajectory id 无交集（`sanity_check` PASS）  
2) **dt 语义明确**：dt-fixed 数据集 dt=30s（全量检查 PASS）  
3) **strict 产物合同齐全**：`data_stats.json` 含 `source`；`nav_field.npz` 含 `metadata.source_split=train`  
4) **坐标范围合理**：pos 在 grid 范围内（抽样 PASS）  
5) **nav_field 对齐不过分离谱**：`mean|cos|` 在可接受范围（考虑道路双向流，`mean_cos` 低并不一定是 bug）

参考：`src/utils/sanity_check.py`，`docs/PHASE_B_RESULTS.md#2`

---

## 3) 关键证据链（从 v1.0 shrinkage 到 v1.1 residual）

### 3.1 v1.0：diffusion/physics 的系统性 shrinkage（宏观偏小）

在 dt30 quick eval（test, 320 条 condition）中：

- Baseline（K=1）：宏观幅度接近 GT（但确定性，多样性=0）
- Diffusion（K=20）：Rog 约为 GT 的 0.58×（显著收缩）
- Physics（K=20）：Rog 约为 GT 的 0.65×（更“稳”但更保守）

> 我们也做过 `vel_scale` 诊断：能对齐速度/路径长度，但会放大微观方向误差（ADE/FDE/DTW/Fréchet 变差），且 MSD10 仍低于 GT，说明瓶颈不只在“尺度”，还在方向持久性/低频结构。

参考：`docs/ROOT_CAUSE_ANALYSIS.md#4`

---

### 3.2 v1.1：Residual decomposition（prior + residual）是结构性修复

在 fast test（test, K=10）上：

- **Residual Diffusion（data-only）**：speed/Rog 基本恢复到接近 1（宏观尺度恢复）  
- **Residual Physics**：micro 更强，但 macro 仍略保守（MSD10 仍偏低）→ 指向 nav\_field mean-field tether

参考：`docs/RESIDUAL_DIFFUSION.md#4.2`，`docs/PHASE_B_RESULTS.md#7`

---

## 4) 已证伪/低 ROI 的路线（不再烧算力）

- **Temperature/噪声强度**：只能加 jitter，不能稳定抬升净位移（已被外部 review + 我们实验共同证伪）。
- **Naive Macro Loss（含多轮改造/门控）**：fine-tune 或从头训在当前注入方式下收益极低；容易进入“micro 更好、macro 更差”的 safe-play 局部最优。
- **`diff_steps=50` 当作严谨结论**：会引入分布漂移，只能用于粗筛，不能替代 ds100 的结论。

---

## 5) 当前真正待解的问题（希望专家批判性 review）

1) **Physics residual 仍略保守（MSD10/Rog 偏低）**  
   - `nav_patch_channel2=speed` vs `count` 差异很小 → “速度通道限速”不是唯一原因。  
   - `nav_emb_scale` 是弱旋钮（1% 量级）。  
   - 我们怀疑核心在：**nav\_field 作为 mean-field prior 的注入方式（concat 到 global cond）过于刚性，产生 tether。**

2) **收敛性陷阱：继续训练会 micro 变好但 macro 变差**  
   - 这意味着训练目标在“求稳”方向有内在偏置，需要结构/注入层面更强的机制来避免保守化。

3) **Prior 的 low-displacement dominance 已被有效缓解，但仍需守住**  
   - deterministic prior 若用纯 MSE，确实会“胆小”；我们验证了允许 `w>1` 的 `disp_weight=clip` 能显著抬升 prior 的宏观尺度（甚至略超 GT），这为 residual 降负。

---

## 6) 我们拟定的下一步（KISS，20 分钟级证伪）

**主线：先动 conditioning 注入，而不是继续做宏观损失/λ sweep。**

我们计划做一个最小结构改动（不引入新任务/不改采样分布）：

- **NavGate（learnable gating）**：对 `nav_emb` 加一个可学习门控（按 cond 或 cond+obs 预测 gate），让模型自己学“什么时候该信 nav、什么时候该弱化 tether”。  
- 对照实验：gate on/off，其它超参不变；评估口径固定为 `K=1,B=50` 快筛 → `K=10,B=200,ds100` 确认。

---

## 7) 我们希望专家给的建议（最需要讨论的 4 个问题）

1) **conditioning 怎么注入 mean-field prior 更合理？**  
   - concat / FiLM / ControlNet / attention 哪种最稳？  
   - 是否建议“方向信息”与“置信度/密度信息”分路注入（避免 tether）？

2) **如何抑制 physics residual 的保守化收敛陷阱？**  
   - 纯 early-stopping 只能权宜；是否有更 principled 的训练策略（而非宏观损失强推）？

3) **采样/评估加速（不改分布）**  
   - 是否推荐 DPM-Solver/EDM 的 ODE 采样器替换 DDPM loop，以降低评估成本而不改变目标分布？

4) **论文叙事/主指标**  
   - 我们计划以 best-of-K + 分布指标 + MSD/Rog 为主，ADE\_mean 放附录；这个取舍是否合理？

---

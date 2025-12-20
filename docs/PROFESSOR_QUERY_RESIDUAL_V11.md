Implementation Plan, Task List and Thought in Chinese：本文件是一份“可直接改写成邮件”的教授咨询摘要，聚焦 Phase B（dt=30s）在 v1.0 收缩与 v1.1 residual 修复后的最新现状、证据链与待拍板问题。

# 教授咨询摘要：Phase B（dt=30s）从“宏观收缩”到 Residual 修复后的下一步

教授您好，

我们在 dt-fixed=30s 的严格协议下完成了 Phase B 的主线验证，并对 v1.0 的“宏观收缩”做了排雷与结构性修复（v1.1 residual）。下面是我们当前最硬的事实证据与仍需要您拍板的关键问题。

---

## 0) 一句话结论（当前状态）

- **v1.0**：Diffusion/Physics Diffusion 的 best-of-K 能力成立，但典型样本存在稳定的 **宏观收缩**（MSD/Rog 偏小）。  
- **v1.1 Residual**：用 deterministic baseline 做 prior 后，**宏观尺度基本恢复**，且 micro best-of-K 仍有增益；但 physics residual 仍略保守，提示 nav_field 作为 mean-field prior 可能带来 tether 效应。

---

## 1) 严格边界（避免争议点）

- **任务定义**：KnownDestination（推理时终点 `d` 为合法条件输入）。  
- **预测形式**：窗口级预测 `obs_len=8, pred_len=12`，输出 future `vel`（语义为 step displacement）。  
- **数据语义**：Phase B 固定 `dt=30s`，并且 `nav_field` 与 `data_stats` 均 **train-only** 生成并记录 source/hash（无泄漏）。

参考：`docs/TASK_DEFINITION.md`、`docs/PHASE_B_RESULTS.md#2`

---

## 2) 已排除的工程问题（我们确认不是实现 bug）

- split overlap / dt 语义 / train-only 产物来源：strict sanity check 已 PASS。  
- normalization mismatch / padding 污染 / 模型容量不足：已逐项排雷（见 `docs/ROOT_CAUSE_ANALYSIS.md#2`）。

---

## 3) v1.0 的关键事实（收缩是稳定现象）

在 dt30 quick eval（test, 320 条 conditions）中：

- Baseline（K=1）：Rog = 5.494（GT=5.247）→ 幅度准但确定性
- Diffusion（K=20）：Rog = 3.056（≈0.58×GT）
- Physics（K=20）：Rog = 3.435（≈0.65×GT）

证据：  
`data/experiments/baseline_b_dt30_eval_quick/metrics.json`  
`data/experiments/diff_b_dt30_eval_quick/metrics.json`  
`data/experiments/physics_b_dt30_eval_quick/metrics.json`

> 我们也用 `vel_scale` 做过诊断：能对齐幅度但会显著放大方向误差（micro 变差），说明瓶颈不仅是“尺度不足”，还包括方向持久性/低频结构缺失。

---

## 4) v1.1 Residual 的关键事实（结构性修复有效）

Residual 思路：冻结 deterministic baseline 作为 prior，Diffusion/Physics 仅学习 residual：
\[
v = v_{\mathrm{prior}} + v_{\mathrm{residual}}
\]

fast test（test, K=10）：

- **Residual Diffusion**：speed 比≈1.016，Rog 比≈1.004，MSD10 比≈0.935  
  证据：`data/experiments/diff_residual_test_fast/metrics.json`
- **Residual Physics**：speed 比≈0.951，Rog 比≈0.945，MSD10 比≈0.864（略保守，但 micro 更好）  
  证据：`data/experiments/phys_residual_test_fast/metrics.json`

解读：Residual decomposition 基本解决“走不动”的尺度问题；剩下的问题集中在 physics conditioning 的保守 tether 与净位移（MSD10）偏低。

---

## 5) 我们真正卡住的问题（希望您批判性 review）

1) **Physics residual 仍略保守（MSD10 偏低）**：我们怀疑 nav_field 的“均值流先验”在提供方向的同时，也把分布拉向更保守的 mean flow（tether）。  
   - 您是否建议改变 nav_field 注入方式（例如只提供方向、不提供幅度；或对速度通道做不同归一化/门控）？

2) **数据分布“低位移窗口占比高”是否在训练中主导了损失？**  
   - 我们倾向用 soft weighting（按 GT 位移大小加权）缓解 low-displacement dominance（优先用 deterministic prior 侧先做）。  
   - 您是否认为应优先重训一个 displacement-aware baseline prior，再做 residual？

3) **评估/采样加速**：我们不希望用改变分布的 `diff_steps` 来“快评”。  
   - 您是否推荐使用 DPM-Solver/EDM 类的严格采样加速（不改目标分布）来降低评估成本？

---

## 6) 我们拟定的下一步（KISS，时间成本可控）

我们计划按以下顺序推进（每一步都可 20 分钟级证伪）：

1) 重训 deterministic baseline prior（加入 displacement-aware weighting），验证其 speed/MSD/Rog 更接近 GT；  
2) 用新 prior 训练 residual physics（先不叠加 macro loss），复核 physics residual 的 MSD10 是否回升；  
3) 仅做少量 conditioning ablation（避免 sweep），锁定 tether 主要来源：
   - `nav_patch_channel2=count`（direction + log-count 置信度，不提供 speed 幅度）
   - `nav_emb_scale` 作为次级旋钮（已观察到收益快速饱和）

若您同意该路线，我们将把每一步的 `metrics.json + samples.npz` 和固定的评估口径整理成可复现实验包。

谢谢您！

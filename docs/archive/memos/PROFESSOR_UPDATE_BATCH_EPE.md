Implementation Plan, Task List and Thought in Chinese：本文件是一封“可直接发给教授”的更新 memo（基于最新 batch-normalized EPE 结果），用于在不增加沟通成本的前提下，把现状、证据与下一步问题说清楚。

# Update Memo：Batch-Normalized EPE Macro Loss 仍未根治“收缩”

教授您好，

> 注：该 memo 对应的是 Phase B（dt30，窗口级）里“macro loss 作为主线修复 shrinkage”的阶段。Phase B v1.1 之后我们把窗口级主线转向 `prior + residual`，并将剩余瓶颈定位为 nav\_field 的注入/保守 tether（详见 `docs/archive/memos/PROFESSOR_QUERY_RESIDUAL_V11.md`）。而当前 trip-level 主线已转向 `docs/archive/legacy_shenzhen/PHASE_C_RESULTS.md` 与 `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`。此文档保留为阶段性证据与复盘材料。

我们按您上次的建议把训练期 Macro Loss 从 Rog 切换到 **EPE（端到端位移差）**，并做了：

- **Hard gate**：仅在 `t < 0.5*T`（T=100，threshold=50）施加 macro loss，避免大噪声步导致 jitter shortcut。
- **Batch-level normalization**：避免 per-sample relative 分母过小导致梯度不稳定；实现为 `rog_loss=batch_relative`（语义：用 batch 平均 GT 位移做归一化）。

下面是我们最新的事实结果与困惑点，希望请您判断下一步应优先改哪一处。

---

## 1) 最新实验（val, K=1, num_conditions=51200）

模型：Physics Diffusion（带 nav_field），dt=30s，评估为窗口级 `obs_len=8, pred_len=12`，K=1。

我们在同一配置下扫了 `λ ∈ {0.01, 0.03, 0.06}`（其余保持一致），结果几乎完全一致：

- `λ=0.01`：`pred_speed_mean/gt_speed_mean = 1.548/2.593 = 0.597`
  - `MSD_10 / GT_MSD_10 = 196.49 / 752.51 = 0.261`
  - `Rog / GT_Rog = 4.476 / 8.344 = 0.536`
- `λ=0.03`：几乎相同（差异 < 1e-3 量级）
- `λ=0.06`：几乎相同（差异 < 1e-3 量级）

对应文件（仓库内可复现）：
- `data/experiments/phys_ft_batchEPE_l0.01_t50_lr3e-4_e22_s0_eval_val_k1/metrics.json`
- `data/experiments/phys_ft_batchEPE_l0.03_t50_lr3e-4_e22_s0_eval_val_k1/metrics.json`
- `data/experiments/phys_ft_batchEPE_l0.06_t50_lr3e-4_e22_s0_eval_val_k1/metrics.json`

结论：在该区间内继续扫 λ 意义不大（似乎处在“无效平台区”），模型仍显著“收缩/走不动”。

---

## 2) 与旧配置的对比（同为 val, K=1, 51200）

旧的（非 batch-normalized）macro-epe 配置示例：

- `data/experiments/phys_dt30_macro_epe_l0.05_e20_s0_fix_eval_val_k1/metrics.json`
  - `pred_speed_mean/gt_speed_mean = 1.357/2.593 = 0.523`
  - `MSD_10 / GT_MSD_10 = 160.93/752.51 = 0.214`
  - `Rog / GT_Rog = 3.926/8.344 = 0.470`

相比之下，batch-normalized EPE 的确把宏观幅度“往上拉了一点点”（0.523→0.597），但离 GT 仍很远（依然只有 ~0.6x 的速度、~0.26x 的 MSD10）。

---

## 3) 我们的关键诊断假设（需要您拍板）

我们怀疑 macro loss 信号被“数据分布”稀释了：窗口级样本里可能有大量 **低位移/近静止** 的片段，使得：

- 即便用 batch mean 做归一化，macro loss 仍然主要在“鼓励小位移更小”，导致整体偏保守；
- 或者 macro loss 的有效梯度主要来自少量大位移窗口，但被随机采样稀释，导致 λ 在 0.01–0.06 看不出差异。

因此我们想尝试一个最 KISS 的修正：**只在 GT 位移足够大的窗口上施加 EPE macro loss（或按 GT 位移大小加权）**，例如：

- mask：`||gt_disp|| > τ`
- weight：`w = clamp(||gt_disp|| / mean(||gt_disp||), 0, w_max)`

---

## 4) 想请您确认的 3 个问题（以便我们今晚继续推进）

1. 您是否同意：下一步优先从 **“macro loss 的样本选择/加权”** 入手，而不是继续扫 λ？
2. 阈值/加权应如何设才最稳妥？（我们倾向用 val 上 GT 位移分位数，例如 `τ = p60/p70`）
3. 是否建议进一步改 timestep 采样分布：例如训练时 **更偏向小 t**（让 x0_pred 更可靠），从而放大 macro loss 的有效梯度？

非常感谢！

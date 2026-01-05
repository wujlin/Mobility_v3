# 教授咨询问题整理：训练期 Macro Loss（Rog/MSD）出现“抖动但走不远”的反常现象（历史备忘）

Implementation Plan, Task List and Thought in Chinese：本文档用于把我们当前最硬的事实、已做过的实验与最需要教授判断的关键问题，整理成一份可直接发给教授的咨询材料（尽量自洽、可复现、可讨论）。

> 注：本路线对应 Phase B（dt30 窗口级）里“macro loss 修 shrinkage”的阶段性探索，已触发止损（macro fine-tune 收益 <0.01 且容易引入高频捷径/保守收敛）。当时我们把窗口级主线转为 `prior + residual`（见 `docs/archive/phase_b/RESIDUAL_DIFFUSION.md`），并把剩余问题定位为 nav\_field 注入方式导致的 tether；而当前 trip-level 主线已转向 `docs/archive/legacy_shenzhen/PHASE_C_RESULTS.md` 与 `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`。本文件保留用于回溯与未来 rebuttal/appendix 的背景说明。

---

## 0. 一句话结论（我们现在卡住的点）

我们尝试在训练阶段加入 **pos-space 的宏观约束（Rog Macro Loss）** 来修复 diffusion/physics 的“收缩/方向持久性不足”问题，但最新实验出现了反常现象：  
**预测轨迹的平均速度与路径长度显著高于 GT（更“抖动/绕圈”），但净位移相关指标 Rog/MSD 反而更差（“走不远”），微观指标也明显变差。**  
这提示：当前 macro loss 的实现/施加方式可能在优化上“把能量推向高频抖动”，而没有提升低频的 directional persistence。

---

## 1. 任务与数据边界（我们认为没有争议的前提）

- **任务定义**：KnownDestination（推理时终点 `d` 作为合法条件输入，不属于泄漏）。  
- **预测形式**：窗口级预测 `obs_len=8, pred_len=12`，输出 future `vel`（语义为 **step displacement**）。  
- **论文版数据语义**：dt-fixed=30s（Phase B），避免速度/MSD 标度语义不明确。  
- **无泄漏原则**：`data_stats.json` 与 `nav_field.npz` **仅使用 train split** 估计并记录来源（strict products）。

参考：`docs/TASK_DEFINITION.md`

---

## 2. 我们想解决的核心物理问题（宏观层面）

此前通过 `vel_scale` 诊断，我们确认“单纯的尺度不够”不是根因：  
把预测速度/路径长度 post-hoc 校准到接近 GT 后，宏观幅度（Rog/MSD）可以靠近 GT，但微观误差显著变差，说明真正瓶颈更像是：

- **方向/时间相关性不足（directional persistence 弱）**  
- 轨迹缺少低频结构（持续朝某方向推进），导致净位移被抵消

因此我们尝试：在训练中显式加入 **低频/积分空间（pos-space）的监督**（Rog/MSD 类约束）。

---

## 3. Macro Loss 的实现概述（我们当前做法）

目标：在不做昂贵采样（reverse diffusion K 次）的前提下，在训练时引入宏观约束。

- 训练时随机采样 diffusion timestep `t`
- 前向得到 `x_t`、`ε_pred`
- 通过标准 DDPM 公式回推 `x0_pred`
- 将 `x0_pred`（future vel, normalized）反归一化为 `pred_vel`
- 用 `start_pos + cumsum(pred_vel)` 得到 `pred_pos`
- 计算 `Rog(pred_pos)` 与 `Rog(gt_pos)` 的差异作为正则项

实现入口：`src/training/train_diffusion.py`（参数 `--lambda_rog`、`--rog_warmup_epochs`、`--rog_loss relative/absolute`）

---

## 4. 最新观测证据（mid 评估结果）

说明：
- 下表为 **GPU 采样评估（K=20, diff_steps=100）** 的 mid 版本（`--max_batches 200` 的子集评估）。
- 其中 `val_mid` 与 `test_mid` 不可直接做严格对比，但足以暴露“宏观趋势/异常形态”。

### 4.1 加入 Rog loss 后（val_mid, seed0, epochs=20, warmup=5）

| 模型 | λ | split | ADE_best | FDE_best | Rog / GT_Rog | MSD_10 / GT_MSD_10 | pred_speed_mean / gt | pred_path_len_mean / gt |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Diffusion + Rog | 0.05 | val | 10.08 | 16.28 | 2.88 / 7.29 (=0.40×) | 17.0 / 579.5 (=0.03×) | 4.08 / 2.26 (=1.81×) | 48.98 / 27.11 (=1.81×) |
| Physics + Rog | 0.01 | val | 9.23 | 13.68 | 5.08 / 7.29 (=0.70×) | 136.4 / 579.5 (=0.24×) | 4.36 / 2.26 (=1.93×) | 52.31 / 27.11 (=1.93×) |
| Physics + Rog | 0.10 | val | 9.16 | 13.89 | 4.83 / 7.29 (=0.66×) | 131.4 / 579.5 (=0.23×) | 3.72 / 2.26 (=1.65×) | 44.63 / 27.11 (=1.65×) |

对应文件：
- `data/experiments/diff_dt30_rog_l0.05_val_mid/metrics.json`
- `data/experiments/phys_dt30_rog_l0.01_val_mid/metrics.json`
- `data/experiments/phys_dt30_rog_l0.1_val_mid/metrics.json`

**关键反常点（最想请教授判断）**：
- `pred_speed_mean` 与 `pred_path_len_mean` 明显 **大于** GT（更“动/更抖”）
- 但 `Rog`、`MSD_10` 却远小于 GT（“净位移更差”）
- 直觉上像 **随机游走/绕圈**：总路程变长，但位移被抵消；且微观误差也明显上升

### 4.2 参考：不加 Rog loss（test_mid, e100, seed2）

| 模型 | split | ADE_best | FDE_best | Rog / GT_Rog | MSD_10 / GT_MSD_10 |
|---|---|---:|---:|---:|---:|
| Diffusion（no macro） | test | 3.05 | 4.23 | 3.86 / 6.53 (=0.59×) | 159.3 / 505.7 (=0.32×) |
| Physics（no macro） | test | 2.93 | 4.20 | 3.80 / 6.53 (=0.58×) | 156.5 / 505.7 (=0.31×) |

对应文件：
- `data/experiments/diff_b_dt30_h128_b512_lr1e-3_e100_s2_eval_mid/metrics.json`
- `data/experiments/physics_b_dt30_h128_b512_lr1e-3_e100_s2_eval_mid/metrics.json`

> 备注：这只是“参考形态”，因为 split/epochs 不同；但至少说明：当前 Rog-loss 配置在 diffusion 上出现了比“无 macro”更极端的异常（速度更大但净位移更小）。

---

## 5. 我们的初步诊断假设（欢迎教授直接推翻）

1) **Rog loss 可能把梯度推向高频噪声（jitter）而非低频 drift**
   - 我们当前 macro loss 基于 `x0_pred`（由随机 timestep 的 `ε_pred` 回推），在大 t 时 `x0_pred` 可能非常噪声化；
   - 若在噪声较大的 timesteps 上也强加几何约束，模型可能通过“快速来回抖动”来优化局部损失，但整体方向持久性反而更差。

2) **Rog 指标本身不够“对症”（对净位移的约束不直接）**
   - Rog 是围绕轨迹质心的空间 spread；某些振荡形态可以拥有较小 Rog 但较大 path length；
   - 或者 Rog 对我们想要的 “MSD 随 lag 增长” 低频结构约束太弱，应改用 MSD\_k / 终点位移 / 多尺度约束。

3) **训练不充分：e20 可能还没收敛**
   - 目前 Rog-loss 只跑到 20 epoch；而 baseline 使用 e100；
   - 可能需要更长训练才能看到 macro 约束真正起效（但当前异常形态仍值得优先解释）。

4) **实现层面的风险点**
   - macro loss 在训练时使用“从 x0_pred 积分得到 pos”的代理；但评估使用完整采样得到的 traj，二者可能 mismatch；
   - normalization/integration 的细节可能放大振荡（例如 vel 的尺度/分布偏移）。

---

## 6. 希望教授重点回答的问题（我们最需要决策的地方）

1) **训练期 macro loss 用 one-step `x0_pred` 近似是否足够严谨？**
   - 在 diffusion 的训练里，这是可接受的做法吗？
   - 是否必须对 timestep 做权重：例如仅对小 t 施加约束，或对大 t 降权（避免噪声主导）？

2) **宏观约束选 Rog 还是 MSD（或其它）更合理？**
   - Rog 是 spread；MSD\_k 是位移随 lag 增长；对“方向持久性”哪个更对症？
   - 是否推荐 **多尺度 MSD（k=1/5/10）** 作为更稳定的低频监督？

3) **如何抑制 “抖动绕圈” 而不牺牲生成多样性？**
   - 是否需要额外的平滑正则（acceleration/curvature/jerk）？
   - 有无更“扩散模型友好”的低频正则（例如对低通滤波后的轨迹做 macro loss）？

4) **macro loss 应该作用在“采样轨迹”还是“代理轨迹”上？**
   - 如果代理（x0_pred）有偏差，是否存在折中：例如少量 steps 的短采样/partial unroll？

5) **λ 的典型量级与 warmup 策略**
   - λ 应如何标定（以 diff loss 的数值量级为基准？）  
   - warmup 的启用时机：固定 epoch、还是根据训练损失收敛自动触发更合理？

6) **Physics 模型里 nav_field 与 macro loss 的交互**
   - 在不改 nav_field 注入方式（保持变量隔离）的前提下，macro loss 是否应先只在 diffusion 上验证？
   - 或者 physics 更依赖 macro loss 才能发挥“方向引导”的优势？

7) **论文叙事与指标选择**
   - 如果 macro loss 让 `ADE_mean` 变差但 best-of-K/宏观更好，是否应把 `best-of-K + 分布指标 + Rog/MSD` 作为主表核心（我们倾向是）？

---

## 7. 我们的下一步候选方案（等待教授拍板后执行）

> 原则：KISS + 隔离变量 + 可快速验真。

- 方案 A：**t-weighted macro loss**（只在小 t 施加，或按 `w(t)` 衰减大 t），看是否能消除“抖动绕圈”。
- 方案 B：将 macro loss 从 Rog 改为 **MSD\_10 / 多尺度 MSD**（更直接约束净位移随 lag 增长）。
- 方案 C：在 A/B 基础上加一个极小权重的 **平滑项（acceleration penalty）**，专门打掉高频抖动。
- 方案 D：在验证 A/B/C 前，不改 nav_field（避免混淆变量）；确认有效后再讨论滚动 patch / autoregressive（v2）。

---

## 8. 附：复现命令（教授如需快速定位）

训练（示例）：
```bash
python -m src.training.train_diffusion \
  --model_type diffusion \
  --data_path data/processed_dt30/trajectories/shenzhen_trajectories.h5 \
  --split train --exp_name diff_dt30_rog_l0.05_e20_s0_w5 \
  --hidden_dim 128 --batch_size 2048 --lr 1e-3 --epochs 20 --num_workers 16 --seed 0 \
  --lambda_rog 0.05 --rog_warmup_epochs 5
```

评估（示例）：
```bash
python -m src.training.evaluate \
  --exp_name diff_dt30_rog_l0.05_val_mid \
  --model_type diffusion \
  --data_path data/processed_dt30/trajectories/shenzhen_trajectories.h5 \
  --checkpoint data/experiments/diff_dt30_rog_l0.05_e20_s0_w5/last.pt \
  --split val --batch_size 64 --max_batches 200 --num_workers 0 \
  --num_samples_per_condition 20 --diff_steps 100 --save_samples 0 --seed 0
```

# Rectified Flow (RF) Pilot：Physics Residual 方案（24h Time-box）

> 目标：在不破坏当前 v1 主线（Residual Diffusion + Physics）的前提下，用 **Rectified Flow / Flow Matching** 做一个“可证伪”的快速对照实验，验证它是否能更好地保持 **低频结构（Directional Persistence / MSD）**，并显著提升 **采样效率**（20-step ODE vs 100-step diffusion）。
>
> 约束（PI 已批准）：
> - 只做 **Plan B：Physics Residual RF**（不能做 data-only，否则丢掉 physics-informed 的核心资产）
> - **24 小时止损**：demo 跑不通或指标无趋势 → 立刻回到 diffusion 主线
> - **不在 RF 上做 CFG**（先做 A/B 对照，避免无底洞）

---

## 1) 为什么要做 RF（我们要验证什么）

我们当前的核心瓶颈是 **macro–micro trade-off**：
- 纯 diffusion：macro shrinkage（MSD/Rog 偏低）
- 加宏观 loss：容易出现捷径（jitter）
- residual + physics：macro 可控，但仍会在 Pareto 前沿上摇摆（micro 好看时 macro 下滑）

RF/Flow Matching 的理论动机：
- diffusion 是 **SDE 去噪路径**，路径更“曲折”，低频结构（方向持久性）更容易被高频误差干扰；
- RF 学的是从噪声到数据的 **速度场（ODE）**，路径更“直”，理论上更容易保留低频位移结构；
- 更重要的是：RF 的推理天然是 **ODE 积分**，有机会用 **20 步**达到 diffusion **100 步**的质量（工程落地价值很大）。

我们要验证的不是“RF 一定更好”，而是以下两个可证伪命题：
1) **Validity 命题**：RF 能否在不引入 jitter 的情况下，让 `MSD10_R / Rog_R` 更接近 1？
2) **Efficiency 命题**：RF 的 `solver_steps=20` 是否能达到 diffusion `diff_steps=100` 的相近水平？

---

## 2) 关键设计：必须 Residual（避免单位/尺度陷阱）

PI 指出的关键坑：RF 的目标是 $\dot{x}$，若直接在 full-velocity 上学，`x1` 的尺度大、`x0`(noise) 尺度容易不匹配，训练很容易崩。

我们采用 Residual 形式把问题拆小：

$$
v_{\text{GT}} = v_{\text{prior}} + v_{\text{res}}
$$

- prior（冻结 deterministic baseline）负责 **低频尺度（scale anchor）**
- RF 只学 residual（细节/分叉/随机性）

训练 target（normalized space）：
```
target_res = action_gt_norm - prior_vel_norm
```

推理重建：
```
pred_vel_norm = pred_res_norm + prior_vel_norm
```

这保证了 RF 学的是“小尺度残差”，显著降低 scale mismatch 风险。

---

## 3) 实现状态（Repo 内已落地）

模型：
- `src/models/flow/rectified_flow_model.py`：`RectifiedFlowTrajectoryModel`（UNet1D + cond_encoder）
- `src/models/physics/physics_condition_flow.py`：`PhysicsConditionFlow`（nav_patch → CNNEncoder → nav_emb → concat cond）

训练：
- `src/training/train_flow.py`：支持 `flow/physics_flow`，支持 residual prior，支持位移加权 `--disp_weight clip`，支持 `--rf_noise_sigma_auto`

评估：
- `src/training/evaluate.py`：已支持 `--model_type flow/physics_flow` + residual prior + `--save_all_k`

---

## 4) Pilot 实验协议（A/B 对照 + 止损边界）

### 4.1 对照组（必须固定）

对照对象：Residual Physics Diffusion（我们现有主线）
- `diff_steps=100`
- K=10
- 同一 prior、同一 split、同一 max_batches

RF 组：
- `solver_steps=20`（Euler）
- K=10
- 其余口径完全一致

### 4.2 指标优先级

Primary（Validity）：
- `MSD10_R`、`Rog_R`（越接近 1 越好）

Secondary（Micro）：
- `ADE_best` / `FDE_best`（Best-of-K；越低越好）

Efficiency：
- wall-clock（记录一次 eval 的耗时；RF 目标是显著少于 diffusion）

### 4.3 止损边界（24h time-box）

满足以下任一条件就止损回 diffusion：
- RF 无法稳定训练（loss 不下降/NaN/爆炸）
- RF 在 `solver_steps=20` 下 macro 指标无明显改善趋势（相对 diffusion 不增益）
- RF 需要大量调参才能收敛（违反 KISS）

---

## 5) 可复现实验命令（最小链条）

> 说明：以下命令是“快速证伪/证实趋势”的口径，不追求最终最优。

### 5.1 Train（Fast Check）

```bash
export DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
export NAV=data/processed_dt30/nav_field.npz
export PRIOR=data/experiments/baseline_b_dt30/last.pt

# RF: Physics Residual, 20 epochs, max_batches=200 (快速验证)
python -u -m src.training.train_flow \
  --model_type physics_flow \
  --data_path $DATA --nav_file $NAV --split train \
  --prior_checkpoint $PRIOR \
  --exp_name phys_flow_residual_rf_pilot_e20_mb200_s0 \
  --hidden_dim 128 --batch_size 2048 --lr 1e-3 --epochs 20 \
  --max_batches 200 --num_workers 8 --seed 0 \
  --solver_steps 20 \
  --rf_noise_sigma_auto --rf_noise_sigma_auto_batches 10
```

### 5.2 Eval（Val / K=10, B=200）

```bash
python -u -m src.training.evaluate \
  --exp_name phys_flow_residual_rf_pilot_val_k10 \
  --model_type physics_flow \
  --data_path $DATA --nav_file $NAV \
  --checkpoint data/experiments/phys_flow_residual_rf_pilot_e20_mb200_s0/last.pt \
  --prior_checkpoint $PRIOR \
  --split val --batch_size 256 --num_workers 0 \
  --max_batches 200 --num_samples_per_condition 10 \
  --flow_steps 20 --save_samples 0 --seed 0
```

> 注意：先用 `num_workers=0` 保证稳定；确定无问题后再提到 4/8。

---

## 6) 预期现象（我们希望看到什么）

如果 RF 有效（至少在趋势上）：
- 在 `solver_steps=20` 下，`MSD10_R/Rog_R` 不低于 diffusion@100，且更靠近 1；
- `ADE_best/FDE_best` 可能略差或相近（可接受），但不应出现明显 jitter。

如果 RF 无效：
- macro 指标无改善，或出现不稳定抖动；
- 需要大量 solver/超参调参才能“勉强可用”。

---

## 7) 备注：为什么 RF pilot 不做 CFG

CFG 会引入新的变量（训练 dropout + 推理双前向），且与 RF 的 ODE path/step size 强耦合；
在我们当前目标（判定 RF 是否值得投入）下，CFG 只会降低归因清晰度，违反 KISS。

RF 如果在 pilot 阶段证明“本体有效”，再讨论 CFG-on-RF 才有意义。


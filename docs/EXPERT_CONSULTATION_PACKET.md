# 专家咨询材料（Phase B / dt=30s）— 当前瓶颈与待解问题

> 目的：把我们已经做过的排雷、事实证据、以及真正卡住的问题压缩成一页“可讨论材料”，用于向专家/教授请教。  
> 版本：v3 / Phase B（dt-fixed=30s，strict train-only 数据产物）

---

## 0. 一句话结论（当前最硬的事实）

在 dt=30s 的严格版本里，Diffusion/Physics **不仅存在“幅度收缩”（Rog/MSD 偏小）**，更关键的是存在 **方向/时间相关性不足（directional persistence 弱，导致净位移被抵消）**：  
我们用 `vel_scale` 把速度/路径长度校准到接近 GT 后，宏观幅度（Rog/MSD）显著改善，但微观误差（ADE/FDE/DTW/Fréchet）明显变差，且 `MSD_10` 仍低于 GT（约 0.84–0.87×），说明“只调尺度”无法根治，瓶颈在更深层的时序结构。

---

## 1. 任务定义与严格边界（避免争议点）

- **任务**：KnownDestination（推理时终点 `d` 是合法输入，不是泄漏）。
- **输入/输出**：窗口级预测（`obs_len=8`, `pred_len=12`），输出 future `vel`（**step displacement**）。
- **论文版数据语义**：dt-fixed=30s（Phase B），避免 MSD/速度场语义不明。
- **无泄漏合同**：`data_stats.json` 与 `nav_field.npz` **只用 train split** 估计，并记录 `trajectory_ids_sha256` 等来源信息。

参考：`docs/TASK_DEFINITION.md`

---

## 2. 已完成的排雷验证（我们确认“不是工程错误”）

1) **Split 无重叠**：train/val/test trajectory id 无交集（`sanity_check` PASS）  
2) **dt 语义明确**：dt-fixed 数据集 dt=30s（全量检查 PASS）  
3) **strict 产物合同齐全**：`data_stats.json` 含 `source`；`nav_field.npz` 含 `metadata.source_split=train`  
4) **坐标范围合理**：pos 在 grid 范围内（抽样 PASS）  
5) **nav_field 对齐不过分离谱**：`mean|cos|` 在可接受范围（考虑道路双向流，`mean_cos` 低并不一定是 bug）

参考：`src/utils/sanity_check.py`，`docs/PHASE_B_RESULTS.md#2`

---

## 3. 关键现象（Phase B 的核心矛盾）

### 3.1 Baseline 很强（dt30 的“强 baseline 效应”）

dt-fixed 后运动更平滑，确定性 L2 模型更容易学到条件均值，因此在 `ADE_mean` 上天然占优（这并不否定生成模型，但会改变主表叙事与主指标选择）。

### 3.2 Diffusion/Physics 的“收缩 + 方向抵消”

- 未校准时：Diffusion/Physics 的 `Rog/MSD` 显著低于 GT（生成轨迹“走不动”）。
- `vel_scale` 校准后：宏观幅度靠近 GT，但微观指标恶化；且即使速度接近 GT，`MSD_10` 仍偏小（净位移被抵消）。

参考证据：`docs/ROOT_CAUSE_ANALYSIS.md#4`，以及对应 `data/experiments/*velscale/metrics.json`

---

## 4. `vel_scale` 诊断实验（为什么说瓶颈不在“尺度”）

### 4.1 校准协议（val→test，避免 test 泄漏）

- 在 `val` 上评估（`vel_scale=1.0`，K=1），得到 `pred_speed_mean / gt_speed_mean`
- 计算推荐尺度：`vel_scale = gt_speed_mean / pred_speed_mean`（跨 seed 取 median）

校准结果（seeds=0/1/2）：
- Diffusion：推荐 `vel_scale = 1.6395`
- Physics：推荐 `vel_scale = 1.6804`

工具：`src/utils/calibrate_vel_scale.py`

### 4.2 关键观察

- **宏观幅度显著改善**：`Rog` 从 ~3.8 拉到 ~6.3（接近 GT）  
- **微观误差显著恶化**：`ADE_mean/FDE_mean/DTW_mean/Fréchet_mean` 全面变大  
- **更关键**：速度对齐后，`MSD_10` 仍低于 GT（约 0.84–0.87×）→ 更像“方向抵消/随机游走”，而不是单纯幅度不足

> 同时确认外部 review：**Temperature ≠ Scale**。温度调的是抖动/多样性，不是净位移幅度；用温度“撑大 Rog”通常只会增加 jitter、拉坏 ADE。

---

## 5. 我们认为真正卡住的问题是什么（可被专家挑战/修正）

### 5.1 训练目标的“均值回归/保守化”偏置

扩散模型在高不确定性区域会更倾向“求稳”，对长尾大位移（大速度）覆盖不足，导致典型样本偏短、净位移偏小。

### 5.2 时序相关性不足（Directional Persistence Weak）

轨迹的低频结构（持续朝某方向推进）没有被学出来，导致即使步速不低，方向频繁摆动产生抵消，`MSD_10` 偏小。

### 5.3 Nav Field 的“均值场先验”可能带来额外保守性

nav_field 是局部平均流场，可能提升 best-of-K 上界，但把典型样本拉向更平滑/更慢的均值流，进一步加剧收缩（需要专家判断我们该不该动它；当前策略是先不动，避免混淆变量）。

---

## 6. 我们希望专家给的建议（最需要讨论的 5 个问题）

1) **如何让 diffusion 学到更强的方向持久性？**  
   - 是否推荐加入“低频/积分空间”的损失（pos-space loss）或“方向自相关”正则？哪些做法最稳？

2) **训练级 Macro Loss（Rog/MSD）该怎么做才正确且高效？**  
   - 我们的初步方案：不做昂贵采样，在训练时用同一 forward 得到 `ε_pred`，推回 `x0_pred`，由 `x0_pred` 积分得到 `pos_pred`，计算 `Rog_pred` 与 `Rog_gt` 的相对误差作为正则项（几乎不增加计算）。
   - 关键问题：这种“单步 x0_pred 近似”会不会引入严重 bias？是否需要 warmup/分段启用？

3) **x0/ε/v 参数化是否值得尝试？**  
   - 我们倾向不把它当“玄学救命”，但想听专家建议：在轨迹这种强自相关序列里，v-pred 是否更稳？

4) **Physics 条件（nav_field）目前是否应该改？**  
   - 是否需要从“起点一次性 patch”扩展到“滚动 patch / autoregressive 更新”？  
   - 还是先用 macro loss 把低频结构补齐再动 nav？

5) **论文评估主指标与叙事**  
   - 主表是否应以 `best-of-K` + 分布指标（Fréchet/DTW/Energy Score）为主，而把 `ADE_mean` 作为附录解释“生成模型的均值不占优但覆盖更好”？

---

## 7. 下一步（我们拟定的最小可行修复）

> KISS：先只做一个明确动作，验证方向对不对。

- **动作**：启用训练级 `Rog` Macro Loss（Diffusion/Physics 均支持 `--lambda_rog`），保持 nav_field 不变。  
- **验证**：1 个 seed 跑小规模（例如 10–20 epochs）观察：  
  - Rog/MSD 是否上升并更接近 GT  
  - 微观指标是否不再随着 `vel_scale` 放大而崩坏（方向更稳）

一旦趋势明确，再扩展到 3 seeds + 更大评估规模。

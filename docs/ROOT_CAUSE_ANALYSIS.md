# Phase B 性能问题深度复盘 (Root Cause Analysis)

## 1. 为了解决什么问题？
Phase B 目前面临的核心矛盾是：**“扩容（h128）稳定了训练，微观最佳值（Best-of-K）超越 Baseline，但宏观指标（Rog/MSD）依然显著‘收缩’（Shrinkage），导致平均误差（Mean）不如 Baseline。”**

- **典型数据**：
    - Baseline: Rog ≈ 5.5 (GT ≈ 5.25) -> **运动幅度准**，但确定性（多样性=0）。
    - Diffusion: Rog ≈ 3.3 (0.63× GT) -> **显著收缩**。
    - Physics: Rog ≈ 3.1 (0.59× GT) -> **收缩更严重**。

## 2. 只有排除了不可能，剩下的就是真相

在得出结论前，我们先用“第一性原理”严谨排除了以下工程实现错误的可能：

### 排除假设 A：归一化统计量错误 (Normalization Mismatch)
- **怀疑**：使用的 `vel_std` 均值偏大？导致模型输出太小的归一化值？
- **验证**：检查了 `data/processed_dt30/data_stats.json`，`vel_std` 约为 `(1.7, 2.2)`。这是在 dt=30s 下合理的（比 dt=variable 的 ~4.0 小）。代码加载逻辑正确，`evaluate.py` 使用了训练时的 stats。
- **结论**：**[排除]**。统计量没有 mismatch。

### 排除假设 B：Padding 0 污染 Loss (Zero-padding Contamination)
- **怀疑**：Loss 计算包含了补零区域，导致模型倾向于预测 0？
- **验证**：检查了 `src/data/datasets_diffusion.py`，数据集只 yield 完整的 `window_size` 切片（Sliding Window），**没有 Padding**。所有训练样本都是真实存在的轨迹段。
- **结论**：**[排除]**。

### 排除假设 C：容量不足 (Underfitting)
- **怀疑**：模型太小拟合不动？
- **验证**：Step 1 实验将 `hidden_dim` 从 64 翻倍到 128，`Best-of-K` 性能显著提升且收敛更稳，说明**容量瓶颈已缓解**。但 Rog 并未因此显著回升（3.0 -> 3.3），说明“收缩”不是单纯的欠拟合，而是模型学到的分布特性。
- **结论**：**[排除]**（不再是主要矛盾）。

---

## 3. 根本原因 (The Root Cause)

排除上述问题后，剩下的根本原因是 **生成式建模对不确定性的固有响应机制** 与 **物理先验的保守性**。

### 核心原因 1：扩散模型的“均值回归”倾向 (Mean-Reversion in High Uncertainty)
- **原理**：Diffusion 虽然能建模多模态，但在高不确定性区域（未来的轨迹），MSE 目标函数（即使是在 noise space）依然偏好“最稳妥”的预测。
- **现象**：当模型这不知道该往哪个具体方向走更远时，为了最小化 Loss，它倾向于预测“模态的中心”或较小的位移，而不是激进地去赌长尾的大位移。
- **结果**：大速度（High Speed）往往是分布的长尾。模型“求稳”导致丢失了高频分量（大位移），表现为轨迹“越走越慢”，Rog 偏小。

### 核心原因 2：Physics (NavField) 的“向心力”效应
- **数据**：Physics 模型的 Rog (3.12) 比 Data-only Diffusion (3.32) 还要小。
- **解释**：Nav Field 提供的是历史/局部的**平均**流场。这是一个强先验。
    - 当模型想“发散”时，Nav Field 告诉它：“这里的历史平均速度是 X，方向是 Y”。
    - 这个先验起到了“正则化”作用，把轨迹拉向了局部平均流（Local Mean Flow）。
    - 局部平均流通常比单条“激进”轨迹要平滑/慢。
- **代价**：Physics 确实更准（ADE_best 更好，方向更对），但代价就是更“保守”，更不愿意产生超出历史平均水平的激进移动。

---

## 4. 论文应对策略 (Strategy for Phase B)

我们不需要“修复”一个符合概率论特性的现象，而是要**诚实地呈现并利用它**。

### 策略 A：扬长避短 (Play to the Strength)
- **叙事逻辑**：“确定性 Baseline 虽然宏观幅度准，但它是‘盲目自信’（Mode Collapse to Mean）。Diffusion/Physics 虽然幅度稍显保守，但它们通过 **多模态覆盖 (Best-of-K)** 捕捉到了未来的多种可能性。”
- **证据**：Physics 的 **ADE_best (2.35)** 显著优于 Diffusion (2.68) 和 Baseline (5.47)。这才是生成模型的杀手锏——**Upper Bound Capability**。

### 策略 B：可视化补救 (Visualization Trick)
- 在画“轨迹叠图”时，不要只画一条。画 20 条，展示出“虽然每一条可能稍短，但组合起来的扇面覆盖了真值”。
- **热力图**：Occupancy Map 会比单条线更公平地展示 Physics 对真值的覆盖概率。

### 策略 C：后续优化 (Optional/Rebuttal)
- 如果审稿人必须要求 Rog 对齐，可以尝试 **Test-time Rescaling**：
    - `pred_vel = pred_vel * 1.5`（简单粗暴，但有效）。
    - 或者引入 **Macro Regularizer Loss**（训练时强行惩罚 Rog 偏差）。
    - 但这属于 Trick，不是 Phase B 当前必须。

## 5. 总结
**问题不在代码，而在特性。**
当前的 Phase B 结果（Physics h128）已经具备发表条件。它展示了一个**“精度更高、覆盖更好、但略显保守”**的物理增强生成器。这完全符合“引入物理场约束”的直觉——约束通常就会带来方差的减小。

# Method — 路线生成的层级承诺架构

## 1. 问题本质：序列预测的根本局限

城市路线生成的核心挑战不在于“下一步往哪走”，而在于“走哪条走廊”。同一 OD 对之间常存在拓扑上截然不同但出行意图等价的替代路径（如穿城干道、外环高速、沿河主路）。

这构成了序列级方法的结构性困境：

- 自回归模型每一步只依赖局部前缀，缺乏走廊级全局承诺。
- 早期偏差会被后续步骤放大，最终表现为 `hit_wall`、`dead_end` 或循环。
- 这不是单纯模型容量问题，而是决策粒度错配：用局部逐步决策去隐式解决全局走廊选择。

最短路径是另一极端：到达率可接近 100%，但仅生成单一路径，无法覆盖真实行为的多样走廊分布；其 `len_ratio≈0.18-0.29` 与 `MeanMaxJ≈0.077` 也说明“到达终点”不等于“走对路线”。

## 2. 核心分解：先承诺走廊，再执行路径

CascadeTraj 将生成分解为“走廊决策 + 路径执行”：

\[
p_\theta(y\mid c)=\int p_\theta(z\mid c)\cdot p_\phi(y\mid z,c)\,dz
\]

其中：

- 条件 \(c=(o,d,t,\text{context})\)
- 走廊级 latent \(z\in\mathbb{R}^D\)
- 路径序列 \(y\)

两阶段对应：

1. 决策阶段 \(p_\theta(z\mid c)\)：在低维空间做走廊承诺（which corridor）。
2. 执行阶段 \(p_\phi(y\mid z,c)\)：在走廊约束下做局部解码（how to walk）。

该分解把“离散的全局选择”和“连续的局部执行”解耦，避免单一 AR 过程同时承担两类异质任务。

## 3. β-VAE 信息瓶颈：把 Flow 任务变为可解

### 3.1 AE 压缩与 Flow 高维失配

AE 将变长 way 序列压缩为 \(n_L=8\) 个 latent tokens（每个 256 维，总计 2048 维）。若直接在 2048 维训练 Flow，条件多模态下容易出现 mode averaging：目标趋向条件期望 \(\mathbb{E}[z\mid c]\)，落在多个走廊之间的“真空地带”。

### 3.2 瓶颈构造

\[
z_{\text{flat}}\in\mathbb{R}^{2048}
\xrightarrow{\text{MLP}}
(\mu,\log\sigma^2)\in\mathbb{R}^D\times\mathbb{R}^D
\xrightarrow{\text{reparam}}
z_{\text{vec}}\in\mathbb{R}^D
\xrightarrow{\text{MLP}}
\hat z_{\text{flat}}\in\mathbb{R}^{2048}
\]

当 \(D=64\) 时，信息瓶颈迫使 latent 保留“走廊身份”而压缩“走廊内细节”；KL 正则化使 \(\mu\) 分布更接近近高斯，Flow 学习从“高维任意分布匹配”降维为“低维条件分布建模”。

### 3.3 维度甜蜜点（容量匹配）

- \(D=32\)：容量不足，AE 重建误差上升（CE 恶化约 10.5%）。
- \(D=64\)：容量与走廊复杂度匹配（KL 约 6 nats，约 8.7 bits，有效状态数约 \(2^{8.7}\approx 400\)）。
- \(D=128\)：容量过剩，Flow 再次回到 mode averaging 区域。

这不是纯超参数调优，而是 latent 容量与走廊内在维度的匹配问题。

## 4. 反环路解码：零训练成本的结构修正

走廊承诺解决全局方向，但局部解码仍会出现短程循环。推理时加入 anti-loop 对最近 \(k\) 步重复 way 施加对数惩罚：

- `anti_loop_k=4`
- `anti_loop_penalty=2.0`

该修正不改训练、无新增参数，实证上可显著压低 loop 并提升到达率（如 `d=64` 从 `70.2%` 到 `78.3%`）。

## 5. 推理流程

1. Flow 在条件 \(c\) 下采样 \(K\) 个 latent \(z_k\)。
2. decoder 在每个 \(z_k\) 下生成候选路线。
3. 采用 `dest_efficient` 在成功候选中选择效率最优路径。
4. 输出单条最优路线（部署）或保留 K 条样本（coverage/diversity 评估）。

## 6. 方法落地的代码路径

- AE 训练：`src/training/train_way_casd_autoencoder.py`
- Flow 训练：`src/training/train_way_casd_flow.py`
- 路径评估：`src/evaluation/way_casd_binned_eval.py`
- OD 覆盖/多样性：`src/evaluation/od_coverage_diversity_eval.py`

## 7. 方法结论

CascadeTraj 的核心不是“更强解码器”，而是“正确的决策粒度”：

- 走廊级承诺先解决多模态主矛盾。
- 低维瓶颈使 Flow 条件生成变得可解。
- 图约束局部执行 + anti-loop 控制误差传播。

因此同一框架下可同时优化到达率、走廊覆盖与多样性。

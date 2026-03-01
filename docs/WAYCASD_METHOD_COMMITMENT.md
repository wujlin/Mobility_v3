# Method — 路线生成的层级承诺架构

## 问题的结构暗示了什么样的解法

城市路线生成的核心挑战不在于"下一步往哪走"，而在于"走哪条走廊"。同一 OD 对之间常存在拓扑上截然不同但出行意图等价的替代路径（穿城干道、外环高速、沿河主路）。这种多模态性是结构性的——不同走廊在坐标空间中相隔甚远，但在出行意图空间中等价。

序列级 AR 方法每一步只依赖局部前缀，缺乏走廊级全局承诺。早期偏差被后续步骤放大，最终表现为 hit_wall、dead_end 或循环。这不是模型容量问题，而是决策粒度错配：用局部逐步决策去隐式解决全局走廊选择。

问题的结构本身暗示了正确的分解方式：走廊选择是**离散的、低维的、可条件化的**；走廊内执行是**连续的、局部的、可图约束的**。CascadeTraj 的核心设计原则正是"先承诺走廊，再执行路径"：

$$p_\theta(y\mid c)=\int p_\theta(z\mid c)\cdot p_\phi(y\mid z,c)\,dz$$

决策阶段 $p_\theta(z\mid c)$ 在低维空间做走廊承诺（which corridor）；执行阶段 $p_\phi(y\mid z,c)$ 在走廊约束下做局部解码（how to walk）。两类异质任务被解耦，各自在最适合的粒度上建模。

---

## Way-Level Tokenization：为什么 way 是正确的路线表示粒度

路线表示粒度的选择是方法的第一个关键决策。现有三种选择各有根本局限：

- **Node 级**：一条路线展开为数百个 node，AR 模型的 exposure bias 随步数线性累积，加之 node-level 的 softmax 搜索空间巨大（数万 node），生成过程极易偏离。
- **GPS 坐标级**：连续空间中没有路网拓扑约束，生成路线可能偏离道路（off-road artifacts），需要后处理 map-matching 才能恢复图结构可行性。
- **Way 级**：OSM 的 way 是路网的**语义原子**——一段连续的同质道路（如"沿某条街从第 3 号到第 7 号路口"）。way-level tokenization 自然地把 L ~ 数百步的 node 序列压缩到 M ~ 数十步，且保留了拓扑结构（相邻 way 通过共享 node 连接）。

Way 不是任意的中间粒度。它之所以是正确的抽象层次，是因为：(1) 同一 way 内部的 node 序列几乎是确定性的（沿道路行走），多模态性集中在 way 之间的选择上；(2) way 的邻接关系直接反映路网拓扑，使得图约束解码成为可能；(3) 从 ~35K unique ways（Porto）到 M ~ 20-40 步的路线序列，搜索空间和序列长度同时被有效压缩。

---

## Perceiver 压缩：为什么需要固定长度的 latent 表示

Way 序列是变长的——不同 OD 对的路线长度差异很大。但 Flow Matching 需要在固定维度的空间中建模条件分布。Perceiver 的作用是把变长 way token 序列压缩为固定 $n_L = 8$ 个 latent tokens（每个 256 维）。

为什么是 8？这个数字对应的是"一条城市路线大约有 8 个关键转折点"的先验——从起点出发，经过若干主要路口或道路类型切换，最终到达终点。Perceiver 的 cross-attention 机制让 8 个 learnable query token 从变长 way 序列中提取最关键的信息，本质上是一种自适应的路线摘要。

---

## β-VAE 信息瓶颈：为什么直接在 Perceiver 输出上训练 Flow 会失败

Perceiver 输出的 8 × 256 = 2048 维 latent 空间直接用于 Flow 训练会遭遇 **mode averaging**。原因在于走廊多模态的几何特性：不同走廊对应的 latent 点在高维空间中相距甚远，但它们之间的中间点在路网上**不连通**——与图像不同，两张图的中间值仍是合理的模糊图像，但两条走廊的中间 latent 对应的是路网上不存在的"幽灵路线"。Flow 在高维空间中难以学到跳过这些不连通区域的锐利条件分布，倾向于收敛到条件期望 $\mathbb{E}[z\mid c]$，落在多个走廊之间的"真空地带"。

β-VAE 瓶颈的构造：

$$z_{\text{flat}}\in\mathbb{R}^{2048} \xrightarrow{\text{MLP}} (\mu,\log\sigma^2)\in\mathbb{R}^D\times\mathbb{R}^D \xrightarrow{\text{reparam}} z_{\text{vec}}\in\mathbb{R}^D \xrightarrow{\text{MLP}} \hat z_{\text{flat}}\in\mathbb{R}^{2048}$$

这个瓶颈做了两件关键的事：

1. **信息压缩**：当 $D = 64$ 时，迫使 latent 保留"走廊身份"而压缩"走廊内细节"。哪些信息被保留、哪些被丢弃，由重建损失和 KL 正则化共同决定。
2. **分布正则化**：KL 项使后验均值 $\mu$ 的分布更接近高斯，Flow 的学习目标从"高维任意分布匹配"降维为"低维近高斯分布的条件建模"。

### 维度甜蜜点：容量与走廊内在维度的匹配

维度 $D$ 不是一个可以随意调节的超参数，它必须与走廊空间的内在维度匹配：

- **$D = 32$**：容量不足，AE 重建误差上升（CE 恶化约 10.5%），但走廊的大致方向仍可编码，SR 仅下降 5.3pp。
- **$D = 64$**：容量恰好匹配走廊复杂度。KL ≈ 6 nats（约 8.7 bits），对应有效状态数约 $2^{8.7} \approx 400$。Flow val_loss = 0.435，学习任务可解。
- **$D = 128$**：容量过剩，β-VAE 不再被迫压缩走廊信息，latent 分布更散。Flow 重新面对高维模态间隙，val_loss 恶化，SR 灾难性下降 24.8pp。

衰减的非对称性（32 温和 vs 128 灾难性）本身就是一个重要发现：容量不足时模型优雅退化（丢失走廊内细节但保留大方向），容量过剩时模型结构性崩溃（Flow 重回 mode averaging）。

---

## 约束 AR 解码器：为什么不在全体 way 上做 softmax

走廊承诺解决了"走哪条路"，但局部执行仍需要一个解码器把 latent tokens 展开为具体的 way 序列。这个解码器的设计有一个关键选择：**不在全体 ~35K way 上做 softmax，而是在图邻居候选集（max_candidates=32）上做打分**。

这个约束有三重作用：

1. **保证图结构可行性**：每一步的输出必须是当前位置的图邻居，解码路径天然是路网上的合法路径。
2. **指数压缩搜索空间**：从 35K 降到 32，分类准确率大幅提升，exposure bias 显著降低。
3. **latent tokens 提供走廊级 guidance**：decoder 通过 cross-attention 查询 8 个 latent tokens，获得走廊级方向信息，在局部候选中选择与全局承诺一致的 way。

### Anti-loop 解码：零训练成本的结构修正

走廊承诺解决了全局方向，但局部解码仍会出现短程循环（反复在两三个 way 之间来回）。推理时对最近 $k=4$ 步出现过的 way 施加对数惩罚（penalty=2.0），无需修改训练、无新增参数，实证上从 70.2% 提升到 78.3%（+8.1pp）。

---

## Flow Matching 在 $\mu$ 上的条件生成

Flow 的训练目标是 $\mu \in \mathbb{R}^{64}$（β-VAE 后验均值），而非 $z_{\text{vec}}$。$\mu$ 比随机采样的 $z$ 更稳定——它是后验分布的确定性摘要，不含采样噪声。条件 $c = (o, d, t, \text{context})$ 通过 cross-attention 注入 Flow transformer。

推理时：Flow 采样 K=16 个 $\mu_k$ → 通过 vae_latent_to_tokens 映射回 8×256 latent tokens → decoder 逐条生成候选路线 → dest_efficient 在成功候选中选择效率最优路径（部署），或保留所有 K 条（评估覆盖/多样性）。

---

## 方法的核心不是"更强解码器"

CascadeTraj 的核心贡献不在于任何单个组件的技术新颖性（Perceiver、β-VAE、Flow Matching 都是已有技术），而在于**正确的问题分解**：

- Way-level tokenization 把路线生成从 node 级数百步压缩到 way 级数十步，搜索空间和序列长度同时被有效压缩。
- β-VAE 信息瓶颈对齐 latent 容量与走廊内在维度，使 Flow 的条件生成从不可解变为可解。
- 图约束局部执行 + anti-loop 控制误差传播，保证输出的图结构可行性。

这三层机制的组合使得同一框架下可同时优化到达率、走廊覆盖与多样性——而这是序列级方法无论怎样增强模型容量都无法实现的。

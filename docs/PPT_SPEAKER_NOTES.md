# PPT 中文讲稿（逐页口播稿）

对应 PPT 源文件：`essay/slides.tex`（Overleaf 编译时将该文件设为 Main document）。

说明：
- 本讲稿按“页标题/章节顺序”组织，确保逻辑连续、过渡自然。
- `Outline` 页在 PPT 中会出现多次（总目录 + 每个 Section 切换时自动出现一次），讲稿中已分别给出过渡话术。
- 数值结果均来自 **dt-fixed=30s** 的 Phase B 严格流程（train-only 产品、无泄漏）。部分结果为 quick subset（用于快速对比与诊断）。

---

## 封面（无标题页）

各位老师好，我是 XXX。今天我汇报我们在“已知目的地（KnownDestination）条件下的城市轨迹生成”上的工作。
我们的核心目标是：在保证生成多样性（多模态未来）的同时，让生成轨迹在宏观统计上也符合真实城市出行动力学（比如位移尺度、MSD/Rog 等）。

接下来我会按 Phase 的方式汇报：先讲我们怎么把 pipeline 做严谨（Phase A），再讲模型对比与发现的问题（Phase B v1.0），最后讲我们为解决这个问题做的关键改进（Phase B v1.1）。

---

## Roadmap (Phases)

这一页给出整个工作的主线：
1) **Phase A：Pipeline**。把“dt、数据划分、train-only 产品、sanity check”做成严格可复现，避免任何信息泄漏；
2) **Phase B v1.0：Models + Diagnosis**。在同一严格协议下对比 baseline、diffusion、physics diffusion，并诊断系统性失败模式；
3) **Phase B v1.1：Fix**。在诊断基础上提出 Residual Diffusion，把宏观尺度问题真正解决掉。

过渡：接下来我先把任务定义说清楚，这样后面的指标解释和对比才是公平的。

---

## Task Definition

我们做的是 **KnownDestination 的 OD 条件轨迹生成**：
- 输入：历史窗口 `H=8`（约 4 分钟），条件包含时间特征 + OD（起终点）；
- 输出：未来 `F=12` 步的位移序列（dt-fixed=30s，约 6 分钟）。

评估目标有两条线：
1) **Coverage（覆盖/多样性）**：best-of-K 的 ADE/FDE，外加 Fréchet、DTW 这类形状/分布相关指标；
2) **Physical realism（物理一致性）**：宏观动力学统计——MSD 曲线与 Rog。

强调：确定性 baseline 天然只输出一条轨迹（K=1）；而 diffusion/physics diffusion 输出 K 条样本（K=20），所以我们把 best-of-K 作为 generative 模型的“能力上界/覆盖能力”指标。

过渡：下面是整份 PPT 的目录，以及每个 phase 之间的关系。

---

## Outline（总目录）

这一页是总目录。我会按顺序：
1) Phase A：先解决“严谨性与可复现”的问题；
2) Phase B v1.0：做模型对比并定位关键失败；
3) Phase B v1.1：围绕失败模式给出最简、可解释、可复现的修复方案。

---

## Outline（进入 Phase A）

这一页目录高亮 Phase A。Phase A 的核心不是刷结果，而是把“后续所有结论成立的前提”搭牢：如果 dt 不固定、nav_field 用了全量数据、split 重叠，后续任何实验都不严谨。

---

## Phase A (from raw data to strict pipeline)

**Phase A 与“之前阶段”的关系**：我们从原始 GPS（采样间隔不规则）出发，必须先做一个严格的、可复现的预处理流程，否则：
- 速度/位移语义不一致；
- MSD 指数的时间尺度不明确；
- nav_field 可能泄漏测试集交通模式。

**Phase A 的目标**：消除 ambiguity 与 leakage，具体包括：
1) 固定 dt（dt-fixed=30s）；
2) 先 split 再统计（train-only normalizer/nav_field）；
3) 用 sanity check 把数据产物合同化。

**Phase A 的输出**：dt-fixed 数据集、train/val/test splits、train-only 的 data_stats 与 nav_field、以及 sanity_check 报告。

过渡：下一页用一张图把这个“数据合同”流程讲清楚。

---

## Phase A: Strict Data Contract (KISS)

这张图的重点是“**Split-first**”：
1) Raw GPS → dt-fixed resample：统一时间尺度；
2) 然后 split：train/val/test；
3) 再用 **train split only** 去生成 data_stats（归一化）与 nav_field（均值流先验）；
4) 最后训练与评估。

这个顺序保证了：无论是 physics conditioning 还是 normalizer，都不会在统计意义上“看见”测试集信息，从而让对比公平、结论可复现。

过渡：Phase A 打牢地基后，我们进入 Phase B，回答模型层面的科学问题。

---

## Outline（进入 Phase B v1.0）

目录进入 Phase B v1.0。本阶段的定位是“对比 + 诊断”：
我们要在同一 pipeline 下比较 baseline / diffusion / physics diffusion，并把失败模式讲清楚、讲透彻，为后续修复方案提供依据。

---

## Phase B v1.0 (what changes from Phase A?)

**Phase B v1.0 和 Phase A 的关系**：Phase A 提供严格数据与无泄漏 priors，Phase B 才能做可信的模型对比。

本阶段的核心问题是两句：
1) diffusion 是否能提供更好的多模态覆盖（best-of-K）？
2) 加入 nav_field 这个“物理/均值流先验”是否能进一步提升覆盖与合理性？

我们对比三类模型：
- Baseline：确定性 L2 回归，K=1；
- Diffusion：data-only，K=20；
- Physics diffusion：nav_field conditioning，K=20。

过渡：下面我先解释“Physics”在这里到底是什么，不然后面的讨论会概念不清。

---

## Phase B v1.0: What is “Physics” here? (Nav field prior)

这里的 “Physics” 并不是严格动力学方程，而是一个 **train-only 的 mean-flow prior（导航场）**：
- 只用训练集统计每个栅格的平均位移方向、访问次数、平均速度等；
- 推理时以“最后观测位置”为中心 crop 一个局部 patch，通过 CNN 编码后作为 diffusion 的条件输入；
- 直觉上它提供了局部方向性指导：像一个“城市的局部流场/路网偏好”的近似。

强调两点严谨性：
1) nav_field 只来自 train split，避免泄漏；
2) 它本质是 local prior：更多解决“方向偏好/局部可行性”，不一定自动解决“宏观位移尺度”。

过渡：有了这个定义，我们看 v1.0 的量化结果：micro 指标和 best-of-K 覆盖表现如何。

---

## Phase B v1.0: Micro-Level Performance (coverage)

这一页看的是 micro-level：
- baseline 在 ADE_mean/FDE_mean 上往往最好，这是 L2 回归逼近条件均值的自然结果；
- diffusion/physics diffusion 由于能采样 K 条轨迹，在 **best-of-K**（oracle）上更强，体现“覆盖能力”；
- physics diffusion 的 best-of-K 进一步优于 data-only diffusion，说明 nav_field 对覆盖是有帮助的（更容易采到“对”的形状）。

在讲这页时，我建议一句话总结：  
**v1.0 的 diffusion 系列在“覆盖/多样性”上确实占优势。**

过渡：但我们发现一个系统性问题：宏观尺度不对。下面用 MSD/Rog 直接展示。

---

## Phase B v1.0: Macro-Level Failure (shrinkage)

核心发现：**shrinkage（收缩）**。
尽管 best-of-K 很好，diffusion/physics diffusion 生成的轨迹在宏观统计上“跑不远”：
- MSD 曲线整体偏低；
- Rog 偏低；
直觉上就是：模型倾向于更保守、更低速、更短路程的未来。

这不是单个 case 的偶然，而是在大样本上稳定出现的低频偏差。  
它对我们“物理一致性/城市动力学”的叙事是致命的：覆盖再好，如果宏观动力学不对，就难以解释成“真实城市运动规律”。

过渡：为了让这个问题更直观，我给出地理空间的可视化（Scheme B），先看 baseline，再看 diffusion、physics。

---

## Phase B v1.0: Interpretable Maps (Baseline)

这一页是 baseline 的地理空间展示，左边是轨迹叠加（GT vs baseline），右边是预测密度热力图（并叠 GT 等高线做参考）。

需要说明：这不是 road-level map matching，只是用 bbox 做线性投影；目的不是“严格贴道路”，而是展示城市空间中的 **宏观聚集结构与走廊**。

baseline 的特点是：输出单一均值路径，整体形状相对平滑，但缺乏多样性。

过渡：下面看 diffusion，在同一空间展示下，多样性会更明显，但 shrinkage 也会体现出来。

---

## Phase B v1.0: Interpretable Maps (Diffusion)

diffusion 的空间叠图通常能看到更丰富的可能性（因为它是采样式的）。
但结合上一页的 MSD/Rog，我们要强调：  
**多样性不等于宏观合理**。如果整体尺度偏小，空间上会体现为轨迹束更集中、更短、更不“放得开”。

过渡：下面看 physics diffusion：加入 nav_field 后，方向引导是否会更符合城市结构？

---

## Phase B v1.0: Interpretable Maps (Physics)

physics diffusion 在局部方向性上更像“顺着城市的主流向/走廊”走，因此 best-of-K 往往更好。
但重要的是：在 v1.0 里，**它仍然没有根除 shrinkage**——也就是“方向更像了，但跑得还是不够远”。

过渡：下面用 grid-space 的具体 case 把“覆盖 vs shrinkage”的矛盾看得更清楚。

---

## Phase B v1.0: Qualitative Cases (grid space)

这页每个小窗都是一个不同的 OD 条件案例（GT、baseline、diffusion、physics）。
你可以看到 diffusion/physics 在某些 case 上能采到更接近 GT 的轨迹（覆盖能力），但整体上也存在：
- 轨迹长度/位移不足；
- 有时会过于保守地贴近局部区域。

这页的目的不是挑最漂亮的例子，而是让观众直观看到：  
**我们的问题是系统性“低频尺度偏差”，而不是完全不会走路。**

过渡：接下来给出更“统计化”的诊断图：误差分布 CDF，以及 Rog 的分布。

---

## Phase B v1.0: Distribution Diagnostics

左图是 ADE/FDE 的 CDF：它能展示“整体分布”而不只是均值。  
右图是 Rog 的分布：直接看宏观尺度是否与 GT 对齐。

讲这页建议强调两点：
1) generative 模型的 ADE_mean 不一定赢回归基线，这是正常 trade-off；
2) 但 Rog/宏观分布如果系统性偏低，就说明“动力学不对”，需要结构性修复，而不是调参小修小补。

过渡：我们确实尝试了训练时宏观约束（macro loss），但效果不理想。下一页解释为什么。

---

## Phase B v1.0: Why macro-loss ablation is insufficient

我们尝试在训练中加入宏观约束（例如 EPE / 位移相关 loss），理论上是对症的，但实践上遇到困难：
- diffusion 在大 timestep 时 $x_0^{pred}$ 噪声很大；
- 在噪声代理上强加几何约束，会诱导模型走捷径（高频抖动/jitter）；
- 即使做了 gate 和归一化，仍然在一定区间出现“提升有限、进入 plateau”。

结论：**单纯靠 macro loss 纠偏低频尺度，在当前架构下不稳健**。因此我们需要一个更结构化、更 KISS 的解决方案。

过渡：这就引出 Phase B v1.1：Residual Diffusion。

---

## Outline（进入 Phase B v1.1）

目录进入 Phase B v1.1。本阶段不是继续调参，而是基于失败模式给出一个可以稳定落地的修复：  
把“低频尺度”和“高频随机性”分开建模。

---

## Phase B v1.1 (link from v1.0 to fix)

先把 v1.0 的问题总结成一句话：  
**diffusion 能覆盖，但倾向于向均值回归，导致低位移（shrinkage）。**

v1.1 的目标也一句话：  
在不牺牲多样性的前提下，把 MSD/Rog 等宏观统计拉回到接近 GT 的水平。

我们的方法直觉非常简单：让一个确定性模型负责“走多远”（尺度），让 diffusion 只负责“可能怎么偏离”（多模态残差）。

过渡：下一页给出 Residual Diffusion 的公式与训练/推理流程。

---

## Residual Diffusion (v1.1): Method

核心分解公式：
$$
\mathbf{v}_{1:F} = \mathbf{v}^{prior}_{1:F} + \mathbf{v}^{res}_{1:F}, \quad
\mathbf{v}^{prior}_{1:F} = f_{base}(\mathbf{X},\mathbf{c})
$$

三句话讲清训练与推理：
1) 先训练一个确定性 baseline，当作冻结 prior；
2) diffusion 训练目标变成 residual：$\mathbf{v}^{res}=\mathbf{v}-\mathbf{v}^{prior}$；
3) 推理时采样 residual，再加回 prior 得到完整轨迹。

这相当于把任务拆成“低频结构 + 高频随机扰动”，是生成建模里很经典、也很稳健的做法。

过渡：下一页用 quick validation 结果说明：这个结构性修复是否有效。

---

## Residual Diffusion (v1.1): Quick Validation

这里是同一组条件（val subset）的对照表：
- baseline（prior）K=1；
- residual diffusion K=20。

要点解读：
1) **best-of-K 覆盖显著提升**：ADE_best 从 4.60 降到 2.50，FDE_best 从 7.24 降到 3.05；
2) **宏观统计明显恢复**：Rog/GT 约 0.93，MSD10/GT 约 0.84，speed ratio 约 0.95；
3) **多样性没有塌缩**：ADE_std 非零，说明 K 条样本不是完全一样的轨迹。

强调边界：这是 quick validation（用于确认方向正确），下一步会做更完整的 test 评估与 residual physics。

过渡：下面给两页“地理空间证据”，把 residual diffusion 的效果更直观地展示出来。

---

## Phase B v1.1: Interpretable Maps (Residual Diffusion) --- Overlay

这一页是轨迹叠加图（bbox 投影，不做 road-level map-matching）：
- 左边是 baseline prior（确定性一条）与 GT 的对比；
- 右边是 residual diffusion（v1.1）与 GT 的对比（图里写的 Diffusion 指的是 residual diffusion）。

读图要点：
1) 由于 prior 提供了低频尺度，v1.1 的整体位移/尺度不会像 v1.0 那样明显收缩；
2) residual diffusion 在 prior 周围提供“随机偏离/多模态”，因此覆盖能力更强（与上一页 best-of-K 对应）。

过渡：下一页看密度热力图（Pred heatmap + GT contour），验证空间分布是否更贴近 GT。

---

## Phase B v1.1: Interpretable Maps (Residual Diffusion) --- Density

这一页是密度层面的对比：
- 仍然是左 baseline、右 residual diffusion（v1.1），并叠加 GT 等高线作为参照。

读图要点：
1) prior 决定了主走廊/主要结构；residual 使得分布更“厚”、覆盖更多可能性；
2) 这张图的价值是把“点对点误差”提升到“空间分布匹配”，更符合 generative model 的评价方式。

过渡：最后我总结一下我们真正学到了什么，以及下一步如何把这条线做成 paper-ready。

---

## Residual Diffusion: What we learned

三条结论：
1) shrinkage 本质上是 **低频学习偏差**（mean reversion），不是简单的“温度/尺度校准”能解决；
2) residual decomposition 是一个 **原理正确且工程稳健** 的修复：把尺度交给 prior，把不确定性交给 residual；
3) nav_field/physics conditioning 的角色更适合当“方向路标”：当尺度稳定后，它更可能发挥正向引导而不是成为保守 tether。

过渡：最后一页我用三句话总结各 phase 的贡献，并列出我们下一步要补齐的实验。

---

## Takeaways \& Next Steps

一句话总结每个 phase：
- Phase A：把 pipeline 做严谨（dt-fixed + split-first + train-only products），保证实验可信；
- Phase B v1.0：证明 diffusion 的 best-of-K 覆盖优势，同时定位 shrinkage 失败模式；
- Phase B v1.1：用 residual diffusion 给出结构性修复，宏观统计显著恢复且保留多样性。

下一步（paper-ready）：
1) residual physics diffusion（nav_field + residual prior）；
2) 更强生成 baseline（例如 CVAE 等）；
3) robustness：prior-swap test + 更吸引人的地理可视化（zoom、底图/瓦片、关键区域 case study）。

结束语：谢谢老师，欢迎提问。

---

## References

这一页是参考文献。我主要引用了两类工作：
1) diffusion/score-based 的基础方法；
2) 轨迹预测/生成与物理一致性相关的工作（包括城市数据集来源）。

如果老师对某个模块（例如 nav_field 构建、residual 训练细节、或评估协议）想深入，我可以回到对应的实验与脚本细节进一步展开。

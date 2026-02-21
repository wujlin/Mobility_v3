下面是我按你们这次“路径/轨迹生成的解码范式（decode paradigm）”需求做的、面向 2023–2026 的针对性文献调研结果（偏“路网图上离散节点/路段序列生成”，以及“非 AR 的离散序列生成+约束满足机制”）。我把每篇都按你要的字段整理：年份/venue、解码范式、是否在图/路网上、如何处理结构约束、与 Way‑CASD 的相关性，并且最后单独回答你最关心的“有没有离散图节点序列 non‑AR 且保证图可行性”的结论。

一、路网/路径（graph/road-network constrained）生成：2023–2026 代表性工作

1. GDP（Graph‑Constrained Diffusion for End‑to‑End Path Planning），2024，ICLR
   解码范式：非 AR（diffusion），但为了满足连通性引入了 search（beam search），因此是 hybrid（diffusion + search）。
   是否在图/路网上：是，明确以 road networks 等图为约束做 path generation。([开放评审][1])
   结构约束怎么做：它的扩散过程在 forward/noise 里引入图结构（文中类比 heat conduction / graph-based transition），做 OD 条件化采样；但关键点是——作者明确说“严格连通性（strict connectivity）与 diffusion 的非自回归生成是冲突的”，因此在生成后用 beam search 来“把每步必须相邻”这个硬约束补回来（把模型对每个位置预测的分布当 proposal，再做 beam search 找到连通路径）。
   与 Way‑CASD 的相关性：高。你们现在 AR rollout 误差复合导致 loop、len_ratio 高；GDP 的 framing 基本就是“非 AR 生成整条路径”来规避 error accumulation，但它也暴露了核心难点：只靠 diffusion 的 position-wise 预测很难硬保证邻接约束，最后还是靠 search。
   局限/负面结论价值：非常高。它几乎就是在你们要找的点上给了“为什么难”的明确论断：strict connectivity vs non‑AR diffusion 的张力。

2. DiffPath（Generating Road Network Based Paths by Latent Diffusion），2025（ICLR 2025 under review / OpenReview）
   解码范式：latent diffusion（非 AR 的迭代采样）+ “从连续 latent 回到离散路段序列”的解码。整体属于 non‑AR 迭代式生成。
   是否在图/路网上：是，目标就是 urban road networks 上的 path generation。
   结构约束怎么做：文中提出 “clamping mechanism” 把连续 latent 的输出拉回到“最近的有效路段 embedding”，并且把它作为“topological validity + contextual coherence”的关键机制之一；同时其离散化（reverse mapping）是 factorized 的位置独立 softmax（给定 z0，每个位置各自分类），这在解码形态上是 non‑AR。
   与 Way‑CASD 的相关性：中到高。它是少数直接做“离散路段序列 path generation + diffusion”的工作之一，且提供了“连续空间采样→离散 token”的落地套路（clamp-to-valid-token）。
   主要不确定/局限：从我能检索到的公开段落来看，它强调“topological validity”，但其核心离散化是 position-wise 分类，是否对“相邻必须可达”给出严格保证，需要以论文方法细节为准（有可能是“经验上更可达/更拓扑一致”，但不等于硬约束保证）。这点对你们很关键：如果你们需要“保证每步是前一步的图邻居”，可能仍要额外加 DP/search/投影。

3. Diff‑RNTraj（A Structure‑aware Diffusion Model for Road Network‑constrained Trajectory Generation），2024，IEEE TKDE（arXiv 2024）
   解码范式：diffusion（连续框架）+ decoder，把连续表示映射回“离散 road segment + 连续 moving rate”的混合序列；非 AR 的迭代采样为主。([arXiv][2])
   是否在图/路网上：是，问题定义就叫 road network‑constrained trajectory（每个点是 road segment）。([arXiv][2])
   结构约束怎么做：把“离散路段 + 连续速度/比例”通过预训练嵌入到连续空间，用 continuous diffusion 生成，再用专门 decoder 映射回混合格式，并提出增强 spatial validity 的损失。([arXiv][2])
   与 Way‑CASD 的相关性：中到高。你们同样有“latent → 离散路网 way 序列”的解码瓶颈；它提供了一个可参考的工程路线：先把离散路段序列编码进连续 latent，用 diffusion 在 latent 上采样，再解码回离散。
   局限：它更像“路网约束的轨迹数据增强/生成”，未必聚焦 OD 条件路径规划；另外“空间有效”不必然等价于“每一步严格图邻接可行”。

4. Seed（Bridging Sequence and Diffusion Models for Road Trajectory Generation），2025，WWW 2025 Poster（OpenReview）
   解码范式：hybrid。Transformer 做“沿路段的序列运动模式”（本质上还是 step-by-step 规划/生成倾向），然后 diffusion 在条件下“从噪声恢复下一路段”以增加多样性。([开放评审][3])
   是否在图/路网上：是，明确 road segments。([开放评审][3])
   结构约束怎么做：它把“可行运动模式”更多交给 Transformer 的序列建模（隐式学到邻接转移），diffusion 作为每步的 stochastic recovery 来缓解“纯序列模型多样性不足”。([开放评审][3])
   与 Way‑CASD 的相关性：中。它并没有彻底摆脱 AR（仍围绕“next segment”），因此不直接解决你们“rollout error accumulation”；但它是一个“把 diffusion 当每步去噪/纠错器”的折中范式，可能启发你们做“AR + 局部去噪/局部重采样”的解码器改造。
   值得注意的卖点：作者在摘要里明确把“sequence 模型的 regularity/consistency vs diffusion 的 diversity”作为 tradeoff，并声称提升明显。([开放评审][3])

5. ControlTraj（Topology‑Constrained Diffusion Model for Controllable Trajectory Generation），2024（arXiv 2024）
   解码范式：diffusion（迭代）+ topology constraint encoding（RoadMAE）+ GeoUNet。([ar5iv][4])
   是否在图/路网上：是用 road segments 作为约束输入集 S，但输出是 GPS 点序列（更偏 map-constrained continuous trajectory，而不是严格离散路段序列）。([ar5iv][4])
   结构约束怎么做：用 RoadMAE 编 road topology 约束、GeoUNet 做带地理注意力的去噪网络，并把 road constraints + trip attributes 作为条件。([ar5iv][4])
   与 Way‑CASD 的相关性：中到低（取决于你们最终输出必须是“离散 way 序列”还是允许连续轨迹再 map-match）。它更像“生成在路网附近/沿路网的连续坐标轨迹”，对“离散图邻接硬约束”帮助有限，但对“用结构条件提升可控性/减少跑飞”有参考价值。

6. DiffTraj（Generating GPS Trajectory with Diffusion Probabilistic Model），2023，NeurIPS 2023
   解码范式：diffusion（非 AR 迭代采样）。([NeurIPS 会议录][5])
   是否在图/路网上：否，主要是坐标系 GPS 轨迹生成。([arXiv][6])
   与 Way‑CASD 的相关性：中到低。它是轨迹 diffusion 的“早期标杆”，可以用来对比“扩散式生成避免 AR 误差复合”的基础论据，但它不处理图邻接硬约束。

7. TrajGDM（Simulating human mobility with a trajectory generation framework based on diffusion model），2024，IJGIS
   解码范式：diffusion（不确定性逐步移除），支持离散位置索引形式的轨迹表示。([Giserwang][7])
   是否在图/路网上：不一定（位置索引可以是一般离散 location），不专指 road graph。([Giserwang][7])
   与 Way‑CASD 的相关性：中。它证明“离散 token 轨迹也能做 diffusion 式生成”，但没有“每步必须是图邻居”的硬约束处理。

8. Map2Traj（arXiv:2407.19765），2024，arXiv
   解码范式：diffusion，强调基于 street map 的 zero-shot trajectory generation（到新区域也能生成）。([arXiv][8])
   是否在图/路网上：是（以 street map 作为条件），但仍偏连续轨迹/地图条件化生成而非严格离散路段序列。([arXiv][8])
   与 Way‑CASD 的相关性：中。它的核心贡献是“用地图当条件”，你们如果未来要把 POI/landuse/road embedding 作为条件来做更强的空间可控性，这条线值得看。

9. Traveller（Travel-pattern aware trajectory generation …），2025，Information Sciences（Elsevier）
   解码范式：hybrid（摘要明确是 AR planning + diffusion-based spatial modeling）。([ScienceDirect][9])
   是否在图/路网上：是“人类移动/轨迹”语境，是否严格 road graph 依论文而定。
   与 Way‑CASD 的相关性：中。它代表一种现实取舍：完全抛弃 AR 很难时，就做“AR 负责结构/计划，diffusion 负责空间细节/多样性”。

二、Non‑AR 离散序列生成（2023–2026）及其“约束机制”启示

你们要的“每步 token 必须是图邻居”本质上是一个非常强的局部约束（bigram constraint / FSM 约束）。在非 AR 框架里，难点是：你不是从左到右逐步生成，没法像 AR 那样在每一步直接 mask 掉非法邻居；约束会跨位置耦合（相邻两位必须匹配）。

下面这些工作虽然主要在语言/符号域，但它们提供了“非 AR + 严格约束”的通用推理范式，迁移到路网 token 序列是有希望的（尤其因为“图邻接”就是典型 regular language / finite-state constraint）。

1. SEDD / Score Entropy（Discrete Diffusion Language Modeling by Estimating the Ratios of the Data Distribution），2024（OpenReview：Submitted to ICLR 2024）
   解码范式：masked/discrete diffusion（非 AR），用 score entropy loss 做离散 score matching；强调任意 infilling、速度/质量权衡。([开放评审][10])
   约束机制启示：它证明了“离散扩散模型可以做任意位置填充（arbitrary infilling）”，这对“路径序列长度固定/可变、局部重写”很关键；你们如果要做“iterative refinement 解码器”（而不是一步步 rollout），这类离散 diffusion 的训练/采样框架是基础。([开放评审][10])
   与 Way‑CASD 的相关性：中到高（方法论层面）。它不含图约束，但提供了可扩展的 non‑AR 离散建模范式。

2. MDLM（Simple and Effective Masked Diffusion Language Models），2024，NeurIPS 2024 Poster
   解码范式：masked diffusion（非 AR），主张简单有效的 masked diffusion LM。([arXiv][11])
   约束机制启示：同上，核心价值在“并行 token 生成 + 多轮 refinement”，适合作为你们想替代 AR decoder 的候选“解码形态”。
   与 Way‑CASD 的相关性：中（提供解码范式，不提供图约束）。

3. CDD（Constrained Language Generation with Discrete Diffusion Models / Constrained Discrete Diffusion），2025，NeurIPS 2025（论文/海报页 + arXiv）
   解码范式：离散 diffusion（非 AR）+ 在采样过程中嵌入可微约束优化（projection / 拉格朗日对偶）。([arXiv][12])
   约束机制：把“满足约束”做成每一步 denoise 时的投影/优化过程，而不是生成后过滤。([arXiv][12])
   迁移到路网的直觉：你们可以把“相邻必须可达”编码成约束（比如用邻接矩阵定义的可行性），再在每轮 denoise 后做投影（例如把不满足约束的局部片段投影到最近可行片段）。这类“投影式扩散”是严肃的可行路线（至少在方法论上站得住）。
   与 Way‑CASD 的相关性：高（约束嵌入式 non‑AR 推理）。

4. DINGO（Constrained Inference for Diffusion LLMs），2025，NeurIPS 2025
   解码范式：diffusion LLM（非 AR）+ 动态规划（DP）的 constrained decoding。([NeurIPS][13])
   约束机制：严格满足用户给定的正则表达式（regular expression），并声称“provably distribution-preserving”的 constrained decoding。([NeurIPS][13])
   对你们的关键意义：路网邻接约束本质上就是 regular language（FSM 的状态=当前节点/路段端点，合法转移=图邻接）。DINGO 这类“DP 约束解码”提供了一个非常直接的迁移模板：把“路径语言”定义为正则/自动机，然后在 diffusion 的并行预测下，用 DP 保证最终序列落在可行语言内。
   与 Way‑CASD 的相关性：高（这是我目前看到最接近“non‑AR 但严格保证结构约束”的通用推理武器之一，只是它的应用场景在文本）。

5. Constrained Decoding of Diffusion LLMs with Context‑Free Grammars，2025，arXiv（也有 OpenReview）
   解码范式：diffusion LLM 的 constrained decoding（支持 arbitrary-order generation / multi-region infilling）。([arXiv][14])
   约束机制：把 constrained decoding 归约为 additive infilling 等问题，并给出针对 CFG 的算法。([arXiv][14])
   迁移意义：你们的约束只需要 regular 就够了（比 CFG 简单），因此这条线更多是“证明非 AR diffusion 也能做形式语言约束”的补强证据。

6. SearchDiff（Search‑Augmented Masked Diffusion Models for Constrained Generation），2026‑02，arXiv
   解码范式：masked diffusion（非 AR）+ 在 reverse denoising 过程中嵌入 informed search（training‑free neurosymbolic inference）。([arXiv][15])
   约束机制：每一步 denoise 把模型预测当 proposal set，然后用搜索在约束/目标下选择，从而修改 reverse transition。([arXiv][15])
   对你们的意义：这几乎就是把 GDP 的“diffusion + beam search”推广成通用框架，并且强调“约束/性质满足”。如果你们要在论文里论证“纯 non‑AR 很难，hybrid search 是合理工程解”，这篇可以作为方法论依据之一。

三、Graph‑constrained sequence decoding：CO（TSP/VRP）与分子生成里 non‑AR 怎么保证约束

这块对你们的价值不在“同任务”，而在“同约束类型”：都是强结构约束下的离散生成。结论很一致：很多所谓 non‑AR 方法，最后一步仍依赖某种 decoding/repair/search 才能把“模型输出的软结构”变成“严格可行解”。

A) Combinatorial Optimization（TSP 等）

1. DIFUSCO（Graph‑based Diffusion Solvers for Combinatorial Optimization），2023，NeurIPS 2023
   解码范式：diffusion（非 AR）在图上生成 {0,1} 边/变量解。([NeurIPS 会议录][16])
   是否在图上：是（CO 图）。([NeurIPS 会议录][17])
   约束处理：典型做法是输出“边的概率/heatmap”，再用启发式/局部搜索把它修成可行解（这一范式在后续很多 CO diffusion 论文里反复出现）。
   与 Way‑CASD 的相关性：中到高（范式层面）。你们完全可以借鉴这种“先并行生成全局 soft signal，再用算法把它投影到可行路径空间”的思路，只不过你们的可行空间是“图上连通 walk/path”。

2. DEITSP（An Efficient Diffusion‑based Non‑Autoregressive Solver for TSP），2025，KDD 2025
   解码范式：NAR diffusion（甚至做 one‑step diffusion + 迭代加噪/去噪调度），但最终仍需要 greedy decoding + 2‑opt 等把 heatmap 变成可行 tour。([arXiv][18])
   约束处理：论文明确说“模型生成 adjacency heatmap 不能保证满足 TSP 约束，因此需要专门 decoding 策略产出可行解”。([arXiv][18])
   与 Way‑CASD 的相关性：高（非常像你们的问题结构）。把它类比到路网：你们可以让 non‑AR 模型输出“每个位置的节点分布 / 每条边的得分”，然后用 DP（类似 Viterbi / shortest-path-on-lattice）解出严格相邻可达的路径。

3. DISCO，2024/2025（OpenReview：ICLR 2025 投稿）
   解码范式：diffusion solver（非 AR）+ 限制采样空间/解析求解来提速。([开放评审][19])
   与 Way‑CASD 的相关性：中。它更偏“让 diffusion 推理更快/更可扩展”，你们如果担心 diffusion steps 太多影响 latency，这类工作值得参考。

4. MaskCO，2026（OpenReview）
   解码范式：masked generation（非 AR/迭代 refinement），推理时 mask‑and‑reconstruct，像局部搜索一样逐步改进解。([开放评审][20])
   约束处理：OpenReview 摘要级信息显示它通过“解的局部遮盖与重建”实现 refinement，并主张跨多种 CO 问题有效。([开放评审][20])
   与 Way‑CASD 的相关性：中到高（解码形态非常像你们想要的“非 AR 迭代修正”，特别适合解决 AR rollout 的 error accumulation）。如果把“路径”看成一个待优化的结构，这种 iterative refinement 可能天然比单次生成更稳。

B) 分子/图生成里的“硬约束保证”路线（对你们的启示：projection/operator 是正道）

1. ConStruct（Generative Modelling of Structurally Constrained Graphs），2024，NeurIPS 2024
   解码范式：graph discrete diffusion（非 AR）+ projector/operator，把生成过程始终限制在满足特定硬约束的图集合内（例如 planar、acyclic）。([arXiv][21])
   约束处理：通过 edge‑absorbing noise model + projector operator，保证 forward/reverse 全轨迹都不离开可行域。([arXiv][21])
   对你们的意义：这基本是“硬约束 diffusion”最像样的一条路。你们的“可行域”不是“满足平面性/无环的图”，而是“满足相邻可达的序列（正则语言）”。如果你们能为序列构造类似 projector（把任意序列投影到最近可行 walk/path），就有希望做到“非 AR + 严格可行”。

2. CoCoGraph（collaborative constrained graph diffusion，molecule validity），2025，arXiv
   解码范式：constrained discrete diffusion，主张生成分子并保证化学有效性（valence constraints）。([arXiv][22])
   约束处理：用特定机制（文中提 double edge swapping）来 enforce valence。([arXiv][22])
   对你们的意义：再次印证“要保证硬约束，往往需要在采样/编辑操作层面嵌入约束保持机制”，而不仅是训练一个更强网络。

3. CDGS（Conditional Diffusion Based on Discrete Graph Structures for molecular graph generation），2023，AAAI 2023
   解码范式：离散图结构上的 conditional diffusion。([AAAI Journals][23])
   意义：给你们补一个“图上离散 diffusion”在 AAAI 这条线的代表性入口。

4. SoftMol / SoftBD（A Block‑Diffusion Perspective on Molecular Generation），2026，arXiv
   解码范式：block diffusion（局部双向 diffusion）+ autoregressive 的 hybrid，并强调“在结构约束下”的生成。([arXiv][24])
   对你们的意义：当硬约束太强时，hybrid 往往是现实选择：局部并行/双向 refinement + 少量 AR 结构化生成。

四、你们最关心的结论：有没有“离散图节点序列 non‑AR 且保证图可行性”的先例？

如果把问题限定得非常严格：
(1) 输出是路网图上的离散节点/边序列；(2) 解码不是 AR rollout（而是 non‑AR / diffusion / masked refinement）；(3) 生成出来的序列严格满足“相邻两步必须是图邻居”（硬保证，不靠事后过滤碰运气）——

我在 2023–2026 的检索里看到的情况是：

* 在“路网路径生成”任务本身，最接近的是 GDP（ICLR 2024），但它明确承认 strict connectivity 与 diffusion 的 non‑AR 本性冲突，最终用 beam search 把连通性硬约束补上，所以它是“diffusion + search”的 hybrid，而不是纯 non‑AR 且原生保证。
* DiffPath（ICLR 2025 under review）提出 clamping 等机制来追求 topological validity，但从我能直接引用到的段落看，它的离散化是 factorized position-wise 分类 + clamping 到有效 token embedding，是否对“每步邻接可达”给出严格保证并不清晰（更像是强启发/强归纳偏置）。
* Seed/Traveller 这种“序列模型 + diffusion”路线，本质仍围绕 next-step（AR flavor）来保证运动合理性，因此不满足你们要的“non‑AR 解码范式替代 AR”。([开放评审][3])
* 真正“严格保证约束”的非 AR 推理范式，目前我看到更成熟的来自“diffusion LLM 的 constrained decoding”，比如 DINGO 用 DP 严格满足 regex（正则约束）。路网邻接约束是正则语言/FSM 约束，因此从理论与算法形态上是可迁移的，但它们还没有在“路网离散路径生成”这个任务上形成公认的标准做法。([NeurIPS][13])



[1]: https://openreview.net/forum?id=vuK8MhVtuu&noteId=YaXmQyJ5Kl "https://openreview.net/forum?id=vuK8MhVtuu&noteId=YaXmQyJ5Kl"
[2]: https://arxiv.org/abs/2402.07369 "https://arxiv.org/abs/2402.07369"
[3]: https://openreview.net/forum?id=L6I2KSXiPN&referrer=%5Bthe+profile+of+Xuan+Rao%5D%28%2Fprofile%3Fid%3D~Xuan_Rao2%29 "https://openreview.net/forum?id=L6I2KSXiPN&referrer=%5Bthe+profile+of+Xuan+Rao%5D%28%2Fprofile%3Fid%3D~Xuan_Rao2%29"
[4]: https://ar5iv.org/pdf/2404.15380 "[2404.15380] ControlTraj: Controllable Trajectory Generation with Topology-Constrained Diffusion Model"
[5]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd9b4a28fb9eebe0430c3312a4898a41-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd9b4a28fb9eebe0430c3312a4898a41-Abstract-Conference.html"
[6]: https://arxiv.org/abs/2304.11582 "https://arxiv.org/abs/2304.11582"
[7]: https://giserwang.github.io/papers/IJGIS-2024-01.pdf "https://giserwang.github.io/papers/IJGIS-2024-01.pdf"
[8]: https://arxiv.org/pdf/2407.19765 "https://arxiv.org/pdf/2407.19765"
[9]: https://www.sciencedirect.com/science/article/abs/pii/S1566253525008280 "https://www.sciencedirect.com/science/article/abs/pii/S1566253525008280"
[10]: https://openreview.net/forum?id=71mqtQdKB9 "https://openreview.net/forum?id=71mqtQdKB9"
[11]: https://arxiv.org/pdf/2406.07524 "https://arxiv.org/pdf/2406.07524"
[12]: https://arxiv.org/html/2503.09790v1 "https://arxiv.org/html/2503.09790v1"
[13]: https://neurips.cc/virtual/2025/poster/118623 "https://neurips.cc/virtual/2025/poster/118623"
[14]: https://arxiv.org/abs/2508.10111 "https://arxiv.org/abs/2508.10111"
[15]: https://arxiv.org/abs/2602.02727 "https://arxiv.org/abs/2602.02727"
[16]: https://proceedings.neurips.cc/paper_files/paper/2023/file/0ba520d93c3df592c83a611961314c98-Paper-Conference.pdf "https://proceedings.neurips.cc/paper_files/paper/2023/file/0ba520d93c3df592c83a611961314c98-Paper-Conference.pdf"
[17]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html"
[18]: https://arxiv.org/html/2501.13767v1 "https://arxiv.org/html/2501.13767v1"
[19]: https://openreview.net/forum?id=6JDpWJrjyK "https://openreview.net/forum?id=6JDpWJrjyK"
[20]: https://openreview.net/forum?id=psUjNnLhl9 "https://openreview.net/forum?id=psUjNnLhl9"
[21]: https://arxiv.org/abs/2406.17341 "https://arxiv.org/abs/2406.17341"
[22]: https://arxiv.org/html/2505.16365v1 "https://arxiv.org/html/2505.16365v1"
[23]: https://ojs.aaai.org/index.php/AAAI/article/view/25549 "https://ojs.aaai.org/index.php/AAAI/article/view/25549"
[24]: https://arxiv.org/html/2601.21964v1 "https://arxiv.org/html/2601.21964v1"

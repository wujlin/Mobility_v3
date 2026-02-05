下面我按你列的 4 个方向把“近几年更有操作性、也更贴近你这个 graph-constrained path generation / AE+AR decoder 设定”的文献脉络梳理一下，并且把最后那个关键问题（“z 编码的是 GT 路径时，rollout training 到底有没有理论依据？”）单独回答清楚。

我会刻意把“哪些方法在你的设定里能落地、需要什么前提、为什么”讲透，因为你这个设定最大的坑，其实不在“要不要 rollout”，而在“rollout 后你用什么 oracle/监督来定义‘正确的下一步’”。

一、Scheduled Sampling / Exposure Bias 的最新进展（以及为什么经典版本常被认为有限）

1. 经典 Scheduled Sampling 的定位与局限
   Scheduled Sampling 最初的做法是：训练时用一定概率把 decoder 的历史输入从 ground-truth token 换成模型自己采样/预测的 token，以缓解 teacher forcing 带来的 train-test discrepancy（exposure bias）。经典论文是 Bengio et al. 2015。([NeurIPS Proceedings][1])
   但它一直有两个“结构性争议”：

* 统计一致性/目标函数不匹配：很多后续讨论指出，按 Scheduled Sampling 那种“喂模型自己生成的 prefix，但 loss 仍按原 GT 序列逐 token 做交叉熵”的方式，会形成一个很奇怪的训练分布，未必对应你真正想优化的序列级目标；并且在理论上可能是“统计不一致”的（常被引用的是 Huszár 的批评，以及大量复述/解释）。([ACL Anthology][2])
* “Exposure bias 是否真的是主要矛盾”：Schmidt 2019 对 exposure bias 的讨论更进一步，强调很多现象更应该从“泛化与分布外 prefix 的行为”去看，而不只是“训练时没喂自己输出”。([arXiv][3])

所以你看到“经典 Scheduled Sampling 效果有限”的结论很常见并不奇怪：它的问题不是“做得不够”，而是“监督信号在 off-trajectory prefix 上到底应不应该还是 GT next-token”这件事本身就不自洽。

2. 近几年更主流的改进方向：从“随机替换输入”转向“动态 oracle / imitation learning 视角”
   你关心 constrained decoding（每步候选集受限）的场景，我建议你重点看“Dynamic Oracle + DAgger / Learning-to-Search”这条线，因为它正面回答了：当 prefix 走偏了，下一步的“正确动作”应该是什么？

* DAgger（Ross et al. 2011）给出一个非常关键的理论框架：用当前策略 roll-in 去采样会访问到的 states，然后让 expert 在这些 states 上给动作标签，迭代聚合数据集训练，从而得到在“自己诱导的状态分布”下也表现好的策略，并给出 no-regret 风格的保证。([Robotics Institute CMU][4])
* 但是 DAgger 的前提是：你得有“能在任意 prefix/state 上给出 expert 动作”的 oracle。也就是说：如果你只有一条 GT 路径，但模型走到图上一个不在 GT 上的节点，你是否还能定义“这一步该走哪条边”？这就是你最后那个关键问题的核心。

这条线在序列生成里最经典、也最贴你问题的落地化是 OCD（Optimal Completion Distillation, ICLR 2019）。它把“动态 oracle”具体化为：给定模型生成的任意 prefix，利用动态规划找到“能使最终 edit distance 最小的所有最优后缀”，再把“下一 token 的目标分布”定义为这些最优后缀的首 token 集合。它强调训练时“总在模型采样的 prefix 上训练”（on-policy prefixes），且监督是“最优补全”而不是死跟 GT。([OpenReview][5])

更近的工作是 2024 的 “Improving Autoregressive Training with Dynamic Oracles”，它系统化地讨论了 Scheduled Sampling/DAgger/动态 oracle 的关系，并针对 span-F1、ROUGE、BLEU 这类难做 exact dynamic oracle 的指标，提出 exact（可分解指标）与 approximate（beam search 近似）dynamic oracle 的算法，还在摘要生成（ROUGE）上展示了 DAgger+动态 oracle 优于 teacher forcing 与 scheduled sampling 的结果。([arXiv][6])

3. “不做 rollouts，但让模型更抗 prefix 噪声”的一类实用改法
   如果你暂时没有强 oracle（或者算 oracle 很贵），另一类更工程化的路线是：保持 MLE/teacher forcing 框架，但显式把 decoder 的历史输入扰动掉，让它学会“在不完美历史下也能继续”。在 VAE/AE 语境里尤其常见：

* TeaForN（EMNLP 2020）提出用 N 个 decoder 叠栈，把上一个 decoder 的输出当下一个 decoder 的输入来训练，从而让参数更新跨多个预测步传播；它强调不需要 scheduled sampling 的 curriculum，也不需要训练时采样整段序列（相比 professor forcing / RL 更稳）。([ACL Anthology][2])
* “word dropout / token dropout / 输入噪声”在 latent-variable 模型里非常常见。NeurIPS 2022 的 AWD（Adversarial Word Dropout）论文很直接地说：他们做过 preliminary 的 scheduled sampling，但发现甚至不如 uniform dropout；并且他们给出在 AR decoder 的 ELBO 下，word dropout 如何改变目标的一个推导（把它和条件 PMI 联系起来），从而提供“为什么这种噪声注入在理论上说得通”的视角。([NeurIPS Proceedings][7])

二、Autoencoder / VAE + AR Decoder 的训练：如何处理 exposure bias？尤其是 z 编码目标序列本身时 rollout 怎么做？

先把你的设定抽象一下：你有 encoder 得到 z（这里 z 是由 GT 路径/序列编码出来的），decoder 是 AR 生成 y = (y1…yT)，训练时通常最大化 log pθ(y|z)（或 ELBO 的 reconstruction 项）。问题在于：训练用 teacher forcing（看见 y< t 的 GT），推理时只能喂自己的 ŷ< t，于是误差累积。

这里要分清两类“分布不匹配”，它们经常被混在一起：

A) prefix 分布不匹配（exposure bias）：训练时的 prefix 来自 GT，推理时 prefix 来自模型。
B) z 分布不匹配（posterior vs prior / 真实 z vs 生成 z）：训练时 z=Enc(GT)，推理时 z 可能来自另一个生成器（prior/diffusion/AR-on-z），或者来自噪声更大的输入。

Rollout 主要解决 A，不直接解决 B。你的系统如果最终要“先生成 z 再 decode”，B 往往比 A 更致命：decoder 可能在 Enc(GT) 的 z 上很强，但在生成器产出的 z 上崩掉——这需要你在训练时就把 decoder 暴露在“更像推理时会见到的 z”上（例如对 z 加噪、或混合来自生成器的 z）。

1. 经典“直接 rollout + 仍用 GT next-token 监督”为何在你设定里危险
   当 z 编码的是整条 GT 路径时，如果你 rollout 生成到某一步已经偏离 GT（比如走到图上另一个节点），你再用 GT 的下一个节点当监督，就等价于在训练中强迫模型学习一个“从错误状态跳回 GT 的动作”，但这个动作在图约束下可能根本不可行（候选集里没有），或者即使可行也未必是“最优纠错”。这就是 scheduled sampling 一直被批评的关键原因：off-trajectory prefix 上的监督不应该盲目等于 GT suffix。([ACL Anthology][2])

2. 解决方式 1：给出“走偏后也定义得出来”的动态 oracle（强烈推荐你优先看）
   如果你的任务评价可以被定义为“与 GT 路径的距离/相似度”（比如 edit distance、路径重合率、到 GT 的投影距离等），那么你其实可以像 OCD 那样：

* roll-in：用当前模型在约束下生成 prefix（graph mask / constrained candidate set 直接应用）；
* oracle：在这个 prefix 上，计算“接下来选哪个 token/节点，能使最终与 GT 的距离最小”；
* distill：用 KL/CE 把模型的 next-step 分布拉向 oracle 的 next-step 分布。

这在理论上比 scheduled sampling 干净得多，因为你监督的不是“GT 的下一步”，而是“从当前 prefix 出发，在你关心的指标下最优的下一步”。OCD 在 edit distance/WER 场景给出了完整、可计算的动态规划和训练目标。([OpenReview][5])
2024 那篇 dynamic oracle 工作则更一般化：它把 dynamic oracle 明确描述为 DAgger 的 expert policy，并给出可分解指标的 exact dynamic oracle、以及对 ROUGE/BLEU 的 beam search 近似 oracle。([arXiv][6])

把这套搬到“z 编码 GT 路径”的 AE 里没有本质障碍：z 只是条件变量；真正决定“rollout 后监督是否合理”的是你有没有动态 oracle（哪怕是近似的）来定义走偏后的正确动作。

3. 解决方式 2：不做显式 oracle，用“噪声注入/对抗 dropout”让 decoder 抗偏
   如果你暂时不想实现 OCD 那类 oracle（比如图上 oracle 计算代价太高，或者你的评价指标很复杂），那就走“让训练时的历史输入更像推理时”的路子：token dropout / word dropout / adversarial word dropout。NeurIPS 2022 AWD 给了一个比较扎实的 ELBO 视角推导，并且明确报告他们的 scheduled sampling 在这个设定里并不占优。([NeurIPS Proceedings][7])

4. constrained decoding 下额外的好消息：候选集受限反而让“动态 oracle”更可做
   在图上，每步候选集合就是邻接点集合；这使得你在 rollout 时永远不会生成非法动作。更重要的是：很多时候你可以在走偏后仍定义 expert：

* 如果目标是“到达某个终点/满足某种代价最小”，expert 可以是 shortest path / A* / DP；
* 如果目标是“贴近 GT 路径”，expert 可以是“使与 GT 的 edit distance 最小”的 OCD 风格 oracle，或其近似版本。([OpenReview][5])

三、路径/轨迹生成里的 rollout 或 RL 微调：有哪些“成功案例”，尤其是 graph-constrained sequential decision？

这里我按“你最关心的两类图约束任务”给例子：一类是“纯图上组合优化/路由”（天然 graph-constrained），一类是“自动驾驶/轨迹预测”（闭环误差累积是核心痛点）。

1. 组合优化/路由（graph-constrained decision）里，rollout/RL 是主流训练范式之一
   这条线的代表作是 Kool et al.（ICLR 2019, Attention Model for routing problems）。它明确把“选下一个节点”当成 policy，训练目标是期望 tour cost，用 REINFORCE，并提出了一个非常贴你问题的点：用“确定性 greedy rollout”的成本作为 baseline（rollout baseline），比学习 critic 更稳定、更高效。([pure.uva.nl][8])
   它的 related work 里也总结了 Bello et al. 2016 用 actor-critic/policy gradient 在 TSP 上做无监督（无最优解标签）训练的思路：用采样到的 tour length 做无偏梯度估计，并使用 mask 来避免重复访问节点。([pure.uva.nl][8])

这类工作回答了你“graph-constrained sequential decision 场景有没有成功 rollout training 案例”：有，而且非常成熟；只不过它们多是“有 reward（cost）定义”的 RL 路线，而不是“只有 GT 路径”的纯 imitation。

2. 自动驾驶/轨迹领域：闭环误差累积的“工程化解决方案”非常多

* ChauffeurNet（2019）是一个很经典的闭环味道很重的 imitation 学习路线：用 logged data 做监督，但训练时会“合成扰动/偏离”（perturbations），让模型学会从偏离状态 recover 回来，缓解 compounding error。([Robotics Proceedings][9])
* 更近的综述（2025，Beyond Behavior Cloning in Autonomous Driving: A Survey）专门把“超越 BC/teacher forcing”的方法做了 taxonomy，核心就是围绕闭环/rollout、干预、仿真与 RL 微调展开。([d1qx31qr3h6wln.cloudfront.net][10])
* NAVSIM（NeurIPS 2024）提出了一个用于驾驶策略评估/训练的 non-reactive simulation 框架，本质上也是为了更好地用“更接近闭环”的方式评估与训练。([NeurIPS Proceedings][11])

3. 更“新”的闭环训练范式（2024–2025）

* RoaD（Rollouts as Demonstrations, 2025）非常直白：把“policy rollouts”当作额外 demonstrations 来训练，并报告在长尾场景、碰撞等指标上带来提升（论文摘要里给了显著的相对改进数字）。([arXiv][12])
* MPA（Model-Based Policy Adaptation, 2025）把闭环训练做成“反事实 rollouts + diffusion policy adapter + Q-value model”的组合，核心也是围绕 rollout 产生的分布来做策略适配。([arXiv][13])
* Doe-1（2024）把驾驶建模成 next-token 生成并用 world model 做闭环生成/评估，这类 work 的共同点是：训练/评估都在追求“让模型在自己生成的未来上保持一致性”。([arXiv][14])

四、Imitation Learning 视角：Teacher Forcing = Behavior Cloning，核心难点是 compounding error；DAgger/GAIL 怎么落到你这里？

1. 你的问题如何“精确对齐”到 IL
   Teacher forcing 的 decoder，本质就是在做行为克隆：学习 π(a_t | s_t)，其中 s_t =（历史 prefix、以及你的条件变量 z），a_t 是下一 token/下一节点。部署时因为 π 有误差，访问到的 s_t 分布偏移，于是误差累积。DAgger 的理论与算法就是为这个问题生的。([Robotics Institute CMU][4])

2. DAgger 在“只有一条 GT 路径”的场景卡在哪？
   它卡在 expert/oracle：如果你不能在“偏离后的状态”给出 expert 动作，DAgger 的前提就不成立。Ross et al. 2011 的 DAgger 是在“能查询 expert 在访问到的 states 上的动作”的前提下推 guarantee 的。([Robotics Institute CMU][4])

3. 序列任务里如何解决“偏离后 expert 动作怎么定义”？动态 oracle 就是答案

* OCD（ICLR 2019）提供了一个特别干净的模板：用 edit distance 定义“最优补全”，从任意 prefix 出发都能用 DP 找到最优后缀集合，并构造 next-token target distribution。([OpenReview][5])
* 2024 的 dynamic oracle 工作更进一步把“动态 oracle = DAgger 所需的 expert”这件事讲得很清楚，并扩展到更多评价指标（可分解指标 exact，ROUGE/BLEU 近似）。([arXiv][6])

4. constrained decoding / action-masked 的 IL：近期也有人专门研究
   你提到“每步候选集受限”，这在 IL 里可以看成 action constraints / action masking。2025 的 Action-Constrained Imitation Learning 这类工作把“动作受限的 imitation”单独拎出来讨论（它的设定和你“图上只能走邻居”很接近）。([arXiv][15])
   如果你还关心“在图结构假设下的 regret/理论保证”，像 Trajectory Graph Learning（NeurIPS 2025）这类也在做“结构化假设 + regret bound”的路线。([NeurIPS][16])

五、关键问题：当 latent code z 编码的就是 GT 路径时，rollout training 到底有没有理论依据？还是架构不适合 rollout？

我把结论先说清楚，然后给你“什么条件下成立、什么条件下不成立、怎么改”。

结论 1：如果你说的 rollout training 是“rollout 用模型前缀，但每步监督仍强行用 GT 的下一 token/节点”，那么它的理论依据非常弱，且在图约束下经常会产生不自洽监督（甚至不可行动作）。这就是 scheduled sampling 一直被批评的那类问题：off-trajectory prefix 上，GT next-token 并不一定是“最优纠错动作”，甚至不一定可行。相关的“统计不一致/目标不匹配”批评在很多地方都被引用与复述。([ACL Anthology][2])

结论 2：rollout 本身没有原罪；只要你能定义“走偏后该怎么做”的 oracle（动态 oracle 或 reward），rollout 在理论上不仅站得住，而且往往正是正确做法。
两条最直接的理论落点是：

* DAgger：要求你能对 rollout 访问到的 states 查询 expert 动作标签，然后给出 no-regret 风格保证。([Robotics Institute CMU][4])
* OCD/动态 oracle distillation：要求你能针对你关心的序列级指标（比如 edit distance/路径相似度）从任意 prefix 计算“最优补全策略”，再把它蒸馏进 AR 模型；OCD 明确强调“总在模型生成的 prefix 上训练”，并用 DP 找最优后缀集合。([OpenReview][5])

把这翻译回你的设定（z 编码 GT 路径）就是：
z=Enc(y*) 不会阻止你 rollout；真正决定“rollout 是否有理论依据”的是：当生成偏离 y* 时，你是否仍能定义一个合理的“纠错监督”。如果可以（OCD 风格、或 shortest-path 风格、或 beam-search 近似 dynamic oracle 风格），那 rollout 非常适合；如果不可以，只能硬贴 y*，那 rollout 反而可能把训练信号搞坏。

结论 3：你这个“z 编码 GT 路径”的 AE 架构，最容易犯的错误其实是把 A) exposure bias 和 B) z 分布不匹配混为一谈。

* rollout/DAgger/OCD 解决的是 A：prefix covariate shift；
* 如果你后续还要训练一个生成器来产出 z（而不是永远用 Enc(GT) 的 z），那你还需要额外手段解决 B（比如对 z 加噪、混合生成器产出的 z 来训练 decoder、或改成非 AR decoder 等），否则你会得到“在 Enc(GT) 的 z 上闭环很稳，但在生成 z 上闭环崩”的现象。这个风险在“latent code 直接编码目标序列”的系统里尤其常见（因为 encoder 太强时 decoder 容易学成一个只在训练分布 z 上工作的解码器）。

给你一个可落地的“判断准则”（你可以直接用它决定要不要做 rollout）：

* 你能否对“偏离后的节点/前缀”定义 oracle？

  * 能：强烈建议做 rollout，优先 OCD/动态 oracle 蒸馏（序列指标）或 DAgger（有规划器/最短路 expert）。([OpenReview][5])
  * 不能：不要做“硬贴 GT next-token”的 rollout；更建议走“输入噪声/word dropout/AWD/TeaForN”这类稳健训练，或者直接换成非 AR/迭代式 decoder，避免把训练变成不自洽的监督。([NeurIPS Proceedings][7])

最后补一句你提到的 constrained decoding：它并不自动解决 exposure bias。它能保证动作合法、缩小分支，从而减缓错误发散，但也可能引入“解码分布偏置”（比如语法约束下的偏置采样问题）。Grammar-Aligned Decoding（2024）非常清楚地指出了现有 grammar-constrained decoding 可能带来显著 bias，并提出 ASAp 去逼近真正的“在语法集合上条件化后的模型分布”。([arXiv][17])

如果你愿意，我下一步可以按你的具体任务把“oracle 怎么定义/怎么近似”更具体地落到两种常见指标上：

1. 路径相似度类（edit distance / LCSS / overlap），用 OCD 风格 DP 做 exact/近似 dynamic oracle；
2. 任务代价类（到终点、代价最小），用 shortest-path/A* 做 expert，从而 DAgger 成立。

[1]: https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf "https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf"
[2]: https://aclanthology.org/2020.emnlp-main.702.pdf "https://aclanthology.org/2020.emnlp-main.702.pdf"
[3]: https://arxiv.org/pdf/1910.00292 "https://arxiv.org/pdf/1910.00292"
[4]: https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf "https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf"
[5]: https://openreview.net/pdf?id=rkMW1hRqKX "https://openreview.net/pdf?id=rkMW1hRqKX"
[6]: https://arxiv.org/pdf/2406.09393 "https://arxiv.org/pdf/2406.09393"
[7]: https://proceedings.neurips.cc/paper_files/paper/2022/file/3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf "https://proceedings.neurips.cc/paper_files/paper/2022/file/3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf"
[8]: https://pure.uva.nl/ws/files/44641651/attention_learn_to_solve_routing_problems_.pdf "https://pure.uva.nl/ws/files/44641651/attention_learn_to_solve_routing_problems_.pdf"
[9]: https://www.roboticsproceedings.org/rss15/p31.pdf "https://www.roboticsproceedings.org/rss15/p31.pdf"
[10]: https://d1qx31qr3h6wln.cloudfront.net/publications/beyond_bc_survey_preprint.pdf "https://d1qx31qr3h6wln.cloudfront.net/publications/beyond_bc_survey_preprint.pdf"
[11]: https://proceedings.neurips.cc/paper_files/paper/2024/file/32768f7faf1995026ef9821c696f3404-Paper-Datasets_and_Benchmarks_Track.pdf "https://proceedings.neurips.cc/paper_files/paper/2024/file/32768f7faf1995026ef9821c696f3404-Paper-Datasets_and_Benchmarks_Track.pdf"
[12]: https://arxiv.org/html/2512.01993 "https://arxiv.org/html/2512.01993"
[13]: https://arxiv.org/html/2511.21584v1 "https://arxiv.org/html/2511.21584v1"
[14]: https://arxiv.org/html/2412.09627v1 "https://arxiv.org/html/2412.09627v1"
[15]: https://arxiv.org/html/2508.14379v1 "https://arxiv.org/html/2508.14379v1"
[16]: https://neurips.cc/virtual/2025/poster/118090 "https://neurips.cc/virtual/2025/poster/118090"
[17]: https://arxiv.org/html/2405.21047v1 "https://arxiv.org/html/2405.21047v1"

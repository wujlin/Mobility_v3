# PPT 中文讲稿（逐页口播稿）

对应 PPT 源文件：`essay/slides.tex`（Overleaf 编译时将该文件设为 Main document）。

说明：
- 本讲稿按“页标题/章节顺序”组织，确保逻辑连续、过渡自然。
- `Outline` 页在 PPT 中会出现多次（总目录 + 每个 Section 切换时自动出现一次），讲稿中给出简短过渡即可。
- 数值与图件均来自 **dt-fixed=30s** 的严格流程（train-only 产品、无泄漏）。密度图使用大样本（例如 N=10k）以避免视觉误判。

---

## 封面（无标题页）

各位老师好，我是 XXX。今天我汇报我们在“已知目的地（KnownDestination）条件下的城市轨迹生成”上的工作。
我们希望模型不仅能生成多样的未来（multi-modal, best-of-K），还要在宏观统计与地理空间上“像真实城市出行”。

这次汇报我会把“我们做对了什么”和“我们没解决什么”明确分开，最后给出一个避免走错路的 pivot 方向。

---

## Roadmap

这一页给出主线：
1) Phase A：先把 pipeline 做严谨（dt-fixed、split-first、train-only 产品、sanity check），保证后面任何结论都可信；  
2) Phase B：展示我们已经做成的部分——physics-informed residual diffusion + CFG，在 OD 对齐、宏观统计与微观多样性上有清晰进展；  
3) Phase C：展示根本性缺陷——Destination Gravity（过早收敛/拓扑坍缩），detour 模态缺失；  
4) Phase D：基于第一性原理与文献调研，pivot 到“trip-level 决策 + 连续执行”的 hierarchical 路线。

过渡：下面先把任务定义说清楚。

---

## Task Definition

我们做的是 KnownDestination 的 OD 条件轨迹生成：
- 输入：历史窗口 `H=8`，条件包含时间特征 + OD（起终点）；
- 输出：未来 `F=12` 步位移序列，dt-fixed=30s（约 6 分钟）。

评估目标（两条线都要看）：
1) 覆盖/多样性：best-of-K 的 ADE/FDE + 分布类指标（Fréchet/DTW 等）；  
2) 有效性/真实性：宏观统计（MSD/Rog）+ 物理统计分布（speed/turn/accel）一致性与违例率（DCV）。

本次汇报的关键点：对于 trip-level，这其实是“多模态路线决策 + 连续执行”的问题，不能只用局部 jitter 来冒充 detour。

---

## Outline（总目录）

目录页：按 Phase A → B → C → D 讲。

---

## Outline（进入 Phase A）

过渡：Phase A 不是刷指标，而是把“实验可信”的前提做成可复现合同。

---

## Phase A (from raw data to strict pipeline)

从原始 GPS（不规则 dt）出发，如果不先固定时间尺度、先 split 再统计，就会出现语义不一致与潜在泄漏，导致后续对比不可信。
本阶段产出：dt-fixed 数据集、train/val/test split、train-only 的 data_stats/nav_field，以及 sanity check。

过渡：下一页用一张图说明 split-first 合同。

---

## Phase A: Strict Data Contract (KISS)

这张图的重点是 split-first：
dt-fixed → split → 用 train-only 生成 data_stats/nav_field → 训练/评估。
这样 physics conditioning 与 normalizer 都不会看见 val/test 信息，保证结论可复现。

过渡：进入 Phase B，先展示我们“已经做成”的部分。

---

## Outline（进入 Phase B）

过渡：Phase B 先讲我们做对了什么（OD 对齐 + 微观 best-of-K + 物理统计），再讲 Phase C 的根本性失败。

---

## Our Method: Physics-Informed Residual Diffusion

这一页是一张方法总览图：
- Prior（确定性）作为 anchor，负责低频尺度/主路径；
- Residual diffusion 只学 residual，负责多模态偏离；
- nav_field 提供局部均值流方向作为条件；CFG 是推理期旋钮。

过渡：先看宏观地理空间上的 OD 对齐（密度）。

---

## Macro: OD alignment in geographic space (Density, N=10k)

这页是 N=10k 的密度图：GT 在左，模型在右（带 GT contour）。
我们要强调的是：在 OD 条件下，模型生成的 occupancy 走廊结构与真实城市高密度走廊是一致的，这证明我们在“宏观空间对齐”上投入的工程是有效的。

过渡：宏观像城市之后，再看微观：best-of-K 是否提供更真实的执行细节。

---

## Micro: Multi-modal local execution (best-of-K)

这页是关键展示：左边 prior anchor，中间/右边 residual diffusion 的多条样本。
它直观说明 diffusion 在 best-of-K 上能生成更多样、也更“物理”的局部形态（比如转弯纹理、速度变化对应的形态差异），相比 seqpred 更接近真实出行的局部随机性。

过渡：用学术界通用语言，我们还要看物理统计分布与 MSD。

---

## Physical Statistics (Validity): Speed/Turn/Accel + MSD

这页展示 speed/turn/accel 的分布与 MSD。
要点是：统计一致性（例如 JSD）在改善，MSD 也更贴近 GT，说明生成轨迹不是纯粹为了刷 ADE，而是在物理统计上更像真实数据。

过渡：但这还不够。接下来讲 Phase C：一个更根本的“拓扑/决策”失败。

---

## Outline（进入 Phase C）

过渡：Phase C 是关键发现：Destination Gravity 解释了为什么一直调 cfg/采样也救不出 detour。

---

## The Defect: Destination Gravity / Premature Convergence

这页是最直观证据：Prior 与 CFG2/CFG3 都出现“被终点吸住”的直冲行为，而 GT 常出现绕路/先逆行上高速等宏观决策。
结论是：当前模型的随机性主要体现在直线附近的抖动/折点，缺少低频的 detour 模态。

过渡：下一页解释为什么这在第一性原理下是“必然”的。

---

## Why CFG cannot create detours (first-principles view)

物理直觉：目的地条件相当于势能深坑；没有中间势垒/锚点，轨迹会沿势能梯度最速下降（直线吸引）。
CFG 是推理期放大梯度增益：会改变尾部/抖动/违例率，但不会凭空生成新的低频拓扑模态。
我们也做了 Oracle 类控制变量实验（workstation）：给中间点或从 K 条里 oracle 选，turn 分布并未恢复，说明 detour 大概率不在 support。

过渡：下一页给一个指标快照，强调“统计更像真 ≠ 拓扑正确”。

---

## Evidence snippet (validity metrics, dt-fixed)

这页用一个 JSD/DCV 的快照说明：统计一致性可以改善，但仍然可能拓扑坍缩。
因此继续在 prior+diffusion+CFG 上扫参不应成为主线，必须 pivot。

过渡：进入 Phase D，给出基于文献的正确问题表述与下一步路线。

---

## Outline（进入 Phase D）

过渡：Phase D 重点是“正确建模对象”——trip-level 决策 + 执行，并对齐 SOTA。

---

## Reframing: trip-level decision + continuous execution

这页是关键重述：
- 决策（macro）：路线拓扑/waypoints（多模态、离散成分强）；  
- 执行（micro）：在给定计划下生成平滑可行的轨迹（连续控制）。  
端到端把两者混在一起，最容易学到的捷径就是“直线到终点 + jitter”。

过渡：下一页对照 SOTA 的两条主流路线。

---

## How SOTA addresses Destination Gravity

SOTA 两条路线：
1) Graph/Map-based diffusion：在路网图上生成，拓扑合法但工程重、数据要求高；  
2) Hierarchical（coarse-to-fine）：先生成 waypoints 再生成细轨迹，map-free 下性价比高、最直接解决 detour。

过渡：最后一页给我们下一步的 KISS 计划，强调“先证伪再工程”避免走错路。

---

## Our next step: hierarchical decision + residual execution (KISS)

计划分三步，且每一步都有可证伪门槛：
1) Oracle upper bound：用 GT waypoints 验证 micro generator 是否具备执行能力；  
2) Learned macro：训练 waypoint predictor（多模态），把 detour 变成显式变量；  
3) Joint system：macro 采样多个 skeleton，micro 并行生成，用 validity gate 评估/选择。  

若 hierarchical 仍失败，再考虑引入 road graph/map（工程量级更大）。

---

## References

这一页是参考文献：diffusion 基础、轨迹预测经典工作、以及 trip-level 的 SOTA（graph-based / hierarchical）。
如需深入某个模块（nav_field、residual 训练、detour 诊断协议），可以回到对应脚本与实验产物展开。


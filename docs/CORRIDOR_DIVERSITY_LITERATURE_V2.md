# CORRIDOR_DIVERSITY：文献调研（定量口径）

> 关注：学术界（尤其是 **route choice / trajectory generation**）如何定义与计算 **corridor**，以及用什么指标衡量 **corridor diversity**。

---

## 1. “Corridor”在不同领域的可计算定义

### 1.1 交通（Route choice）语境：corridor ≈ “重叠路径束 / 子网络组件”
在经典离散选择（Random Utility / Logit family）里，研究对象通常是 **OD 对上的多条候选路径（paths）**。对 *corridor* 的处理主要有两类：

1) **把 corridor 当作“路径之间的重叠结构”**（overlap → correlation）
- 典型做法不是直接显式标注“第 1/2/3 条 corridor”，而是用**重叠度量**（shared links / shared impedance）进入效用或相关结构。
- C-Logit 用 *Commonality Factor*（CF）对“高度重叠”的路径施加惩罚（overlap penalty），以修正 MNL 在重叠路径上概率不合理的问题（Cascetta et al., 1996）。
- Path Size Logit（PSL）用 *Path Size*（PS）刻画一条路径相对选择集的“独特性”（distinctiveness），并把它加到路径效用里（Ben-Akiva & Bierlaire, 1999；Frejinger, 2005）。

2) **把 corridor 当作“行为上可解释的子网络 / 走廊组件”**（subnetwork components）
- Frejinger & Bierlaire 提出以**子网络组件**刻画相关性：把路网中“易识别且行为相关”的主干路/区域定义为 subnetwork components，路径共享这些组件（甚至不完全物理重叠）就被认为存在相关（Frejinger & Bierlaire, 2006；2007）。
- 这种方法在“corridor=主干走廊/主干道路选择”的语义上更接近你关心的 corridor。

补充：**link-based / implicit choice set** 的路线选择建模
- 例如 Recursive Logit（Fosgerau et al., 2013）把决策建模为逐 link 的动态选择，从而不必显式枚举所有路径。此时 corridor 往往体现在**link 序列模式**与其概率质量上，而不是“先提 corridor 再估计”。

**可操作总结（交通领域的可计算 corridor）：**
- corridor 可以被定义为：OD 下的一组路径在某段路网主干（共享 link 或共享 subnetwork component）上高度一致的“路径束”。

---

### 1.2 ML（Trajectory generation / prediction）语境：corridor ≈ “多模态预测的 mode / cluster”
在多模态轨迹预测里，模型输出往往是一组候选未来轨迹（K samples / K modes）。
- “corridor”通常不直接出现为术语，但它对应：**预测分布的一个模式（mode）**，或者在空间上形成一束轨迹的一个**聚类簇（cluster）**。
- 评估上，领域主流长期用 **Best-of-N（minADE/minFDE）**衡量“预测是否覆盖真实未来”，但这类指标本质偏向**准确性/覆盖**而不是“模式之间是否真的不同”。Trajectron++ 总结了常用 ADE/FDE、KDE-NLL、Best-of-N（Salzmann et al., 2020）。
- 为了解决“只追求 minFDE 导致 mode collapse / 伪多样性”，近几年更常见加入**显式 diversity 指标**：例如 Average Pairwise Distance (APD) 衡量一组预测之间的平均两两距离（Yuan & Kitani, 2020；Salzmann et al., 2022）。

**可操作总结（ML 领域的可计算 corridor）：**
- corridor 可以被定义为：在同一历史条件下，模型输出轨迹集合在空间上形成的一个 cluster/mode（例如左转、直行、右转），并用聚类或距离阈值将 K 个样本划分为若干 corridor。

---

## 2. 现有文献用什么指标衡量 corridor diversity？

把“corridor diversity”拆成两个可量化维度会更清晰：

- **(A) 分离度（separation）**：不同 corridor 之间差多远？
- **(B) 有效数量与均衡性（effective number & balance）**：有多少条 corridor？概率/流量是否集中在少数 corridor？

### 2.1 Route choice 文献：重叠惩罚/独特性指标（本质是 corridor separation 的 proxy）

#### (1) Commonality Factor / Overlap penalty（C-Logit）
- C-Logit 在路径效用中加入与其他路径相似度相关的 CF 项（常见形式依赖于两两共享长度 L_{kh} 与路径长度 L_k, L_h），从而对高度重叠路径降低效用/概率。
- 直觉：如果某条路径和很多路径共享长的重叠段，它不是“独立 corridor”，其有效贡献应被折扣。
- 代表：Cascetta et al. (1996)；后续大量 SUE/assignment 论文沿用 CF 思路。

#### (2) Path Size（PS）/ Path Size Logit（PSL）
- PSL 用 PS 衡量路径相对选择集的“独特性”。最常见的 PS 形式是 link-level 加和：

  \[\text{PS}_i = \sum_{a\in i} \frac{l_a}{L_i}\cdot \frac{1}{\sum_{k\in C}\delta_{a,k}}\]

  其中 \(l_a\) 为 link 长度（或阻抗），\(L_i\) 为路径 i 总长度（或总阻抗），\(\delta_{a,k}=1\) 表示路径 k 使用了 link a。

- 直觉：路径 i 的每一段 link，如果被很多候选路径共享，则该段对“独特性”的贡献更小；独占 link 贡献更大。
- 代表：Ben-Akiva & Bierlaire (1999)；Frejinger (2005)。

#### (3) Subnetwork / Error components（corridor-as-component）
- Frejinger & Bierlaire 的 subnetwork 方法把“corridor”更显式化：先定义主干路/区域为组件，然后让共享组件的路径误差项相关。
- corridor diversity 的量化往往体现在：**有多少组件被使用**、以及不同组件路径之间的相关结构（方差/载荷）大小。
- 代表：Frejinger & Bierlaire (2006, 2007)。

> 小结：交通 route choice 的主流指标不是“直接输出 corridor diversity 数值”，而是通过 **重叠度量（CF/PS）**把 overlap→相关性→概率修正。若要得到 corridor diversity，可在其基础上再做 set-level 聚合（例如“平均 PS”“平均两两 overlap”“有效路径数”等）。

---

### 2.2 Trajectory generation 文献：多模态评估中的“多样性指标”

#### (1) Best-of-N（minADE/minFDE）与 KDE-NLL：偏 coverage/likelihood，不等价于 diversity
- Trajectron++ 使用 ADE/FDE、KDE-NLL、Best-of-N（BoN=minADE/minFDE）作为主要指标（Salzmann et al., 2020）。
- BoN 的本质：只要 **K 个样本中有一个接近 GT** 就得高分，因此容易“用少数模式覆盖 GT”，而对“其它样本是否重复”不敏感。

#### (2) APD / FPD：显式衡量样本集合的 spread（更像 corridor separation）
- APD（Average Pairwise Distance）常用来衡量一组预测轨迹两两之间的平均距离，是许多多模态/生成式运动预测论文的标准 diversity 指标之一（Yuan & Kitani, 2020；Salzmann et al., 2022）。
- 一些工作还区分：
  - **APD**：沿时间序列的平均距离（反映整条轨迹差异）
  - **FPD/endpoint diversity**：仅终点差异（对应你提的 FDE diversity）

#### (3) “K-modality / multi-modality metrics”：不仅看 spread，也看“有多少不同模式”
- 典型做法：先把 K 个预测聚类成若干“相似 motion group”，再报告 multi-modal ADE/FDE（MMADE/MMFDE）或 mode-aware 指标，避免单纯 APD 被离群点主导。
- Motron 明确报告 APD 作为 sample diversity，同时沿用 Best-of-N ADE/FDE，并采用 Yuan et al. 的 “Multi-Modal” 评估（Salzmann et al., 2022）。

> 小结：ML 领域把 corridor（mode）当作 cluster，很自然得到 corridor diversity：
> - separation：APD/FPD/Hausdorff 等
> - effective number：cluster 数量、cluster size 分布熵、mode coverage

---

### 2.3 Road network clustering / corridor mining：把 corridor 当作“聚类结果/代表路径”
这一支文献更直接回答“corridor 是什么、怎么提取”：

- **从 GPS 轨迹抽取交通走廊**：
  - Zygouras & Psyllidis（2017）讨论如何从轨迹中抽取 corridors & routes（强调轨迹→路网映射与聚合）。
  - 自行车 GPS 数据中提取 primary corridors 的工作给出了一套可复现的 pipeline，并使用几何距离（如 Hausdorff）等度量轨迹相似性（Jiang et al., 2015/2016）。

- **代表路径/路径聚类（path-level clustering）**：
  - k-paths（VLDB 2020）提出在路网上抽取“代表性路径（representative paths）”，本质是路径聚类与原型提取问题。

> 小结：这类文献最接近你想要的“corridor identification + corridor diversity”。它们常把 corridor diversity 定义为：
> - 聚类得到的 corridor 数量
> - corridor 的流量/样本占比的分布（熵/集中度）
> - corridor 之间的几何或拓扑距离（Hausdorff/Fréchet/编辑距离等）

---

## 3. 他们如何从 GPS 轨迹数据中提取 corridor？（典型 pipeline）

下面把交通 route choice 与 corridor mining 的常见做法统一成一个可复用的流程：

### Step 0：轨迹切分与质量控制
- trip segmentation（按时间/停留点/OD 边界）
- 去噪（异常速度、漂移点）

### Step 1：Map matching（GPS 点 → 路网 link 序列）
- 产物是每条轨迹对应的 **有序 link 序列**（或 node 序列、polyline）。
- 这一步是你后续用 LCS/overlap 的关键：没有 map matching，轨迹距离更倾向于几何距离（Hausdorff/Fréchet）。

### Step 2：定义“候选路径/候选 corridor”
三条常见路线：

1) **OD 内的“唯一 link 序列”直接视为一条 route**
- 适合 GPS 数据密、map matching 可靠的场景。
- corridor 需要再做聚合（否则“微小差异=新 route”会导致 corridor 爆炸）。

2) **聚类（trajectory/path clustering）→ corridor**
- 距离/相似度可选：
  - 几何：Hausdorff / Fréchet / DTW
  - 拓扑：link overlap ratio / edit distance / LCS/LCSS
- 聚类输出：每个 cluster 是一个 corridor；cluster 中的“代表路径”可用 medoid 或 k-paths 类方法抽取。

3) **choice set generation + overlap-aware 模型（route choice）**
- 先生成候选路径集合（k-shortest paths、link elimination 等），再通过 CF/PS 处理 overlap。
- 这条线更偏“解释选择行为”，corridor 通常是隐含在 overlap 结构中的。

### Step 3：计算 corridor diversity（两类汇总口径）

- **Corridor 数量/均衡性（effective number）**
  - corridor 数 K
  - corridor 份额（样本数/流量/概率质量）\(p_1,...,p_K\)
  - 常用汇总：Shannon entropy \(H=-\sum p_k\log p_k\)，有效 corridor 数 \(\exp(H)\)

- **Corridor 分离度（separation）**
  - corridor 间距离：cluster prototype 之间的 Hausdorff / LCS distance / overlap dissimilarity
  - corridor 内紧凑度：intra-cluster distance

---

## 4. 你的 LCS distance 方法与他们的方法差在哪里？

假设你的方法是：
- 先 map match 得到 **link 序列**，
- 再用 **LCS（Longest Common Subsequence）**定义两条路径的相似性/距离（例如 1 - |LCS|/min(|A|,|B|) 或按长度加权），
- 用该距离做 clustering，从而得到 corridors，并据此定义 corridor diversity。

### 4.1 与交通 route choice 的 CF/PS 的差异

| 维度 | CF（C-Logit）/ PS（PSL） | 你的 LCS distance |
|---|---|---|
| 基本对象 | “路径 i 相对选择集 C 的重叠程度/独特性” | “两条路径 i,j 的序列相似性” |
| 是否依赖 choice set | **强依赖**（C 的增删会改变 CF/PS） | 可不依赖（pairwise）；只有在你做 set-level 聚合时才依赖集合 |
| overlap 表达 | 以共享 link 的长度/阻抗加权（更接近交通成本语义） | 以“最长公共子序列”表达；允许跳过段落（非连续共享） |
| 输出形态 | 通常用于效用修正/相关结构（隐式 corridor） | 天然适合做聚类与显式 corridor identification |
| 解释性 | CF/PS 有成熟的离散选择解释（correlation correction） | 更像数据驱动的“路径相似度度量”（解释需你定义） |

关键点：
- **CF/PS 是“模型内的 overlap correction term”**，目标是修正概率与相关性；
- **LCS 是“模型外的距离度量”**，更适合 *先做 corridor identification，再谈 corridor diversity*。

### 4.2 与 Hausdorff/几何距离（corridor mining/trajectory clustering）的差异

- Hausdorff/Fréchet 以几何曲线距离为主：
  - 优点：不一定需要 map matching；能处理非路网场景。
  - 缺点：在平行路/高架上下层/定位误差时，几何距离可能被夸大或混淆。

- LCS（在 link 序列上）更偏拓扑：
  - 优点：对 GPS 噪声更鲁棒（噪声主要在 map matching 前被吸收）；能区分“走了哪条路”。
  - 风险：
    - LCS 允许跳过元素，可能把“同一 corridor 的小绕行”与“完全不同 corridor 的拼接式相似”都视为相似；
    - 如果想强调“连续共享路段”（真正的走廊），可能需要改为 **Longest Common Substring（连续）** 或对 LCS 加入连续性/最小 gap 约束。

### 4.3 与 ML 轨迹预测评估指标（FDE diversity / APD / multi-modality metrics）的差异

- ML 指标多为 **欧氏空间距离**：
  - FDE diversity：只看终点差异（可能忽略中间走廊差异）。
  - APD：看整体 spread，但可能把“同一 corridor 的 lateral jitter”也算作多样性。
  - BoN（minFDE/minADE）：更像 coverage/accuracy，不是 diversity。

- 你的 LCS（基于路网 link 序列）更贴近“corridor”语义：
  - 你测到的是**走廊/路线层面的离散差异**，而不是连续空间的细微偏移。
  - 更适合回答“预测是否覆盖了不同路线选择”（左转 vs 直行 vs 右转），以及“生成的 K 条轨迹是否真的对应不同 corridor”。

---

## 5. 你可以从文献中直接借用的“corridor diversity 定量口径”

为了让你的定义与两大领域都能对齐，建议把 corridor diversity 报告成 **三件套**：

1) **Corridor 数量（K）**：通过聚类得到的 corridor 数。
2) **Corridor 均衡性**：用熵/有效数 \(\exp(H)\) 或 Gini/HHI 描述分布是否集中。
3) **Corridor 分离度**：用 corridor prototype 之间的距离（LCS/Hausdorff）报告“这些 corridor 是否真的不同”。

其中：
- (2) 对齐 route choice 的“流量/概率份额”；
- (3) 对齐 route choice 的 overlap 思维（CF/PS）与 ML 的 APD/FPD。

---

## 参考文献（可作为 Phase 2 深挖清单）

### Route choice / overlap / corridor components
- Cascetta, E., Nuzzolo, A., Russo, F., & Vitetta, A. (1996). *A modified logit route choice model overcoming path overlapping problems* (C-Logit). ISTTT.
- Ben-Akiva, M., & Bierlaire, M. (1999). *Path Size Logit / path size formulation*（见 Handbook of Transportation Science 等）。
- Frejinger, E. (2005). *Route Choice Models with Subpath Components* (STRC 2005). https://www.strc.ch/2005/Frejinger.pdf
- Frejinger, E., & Bierlaire, M. (2006). *Capturing Correlation in Route Choice Models using Subnetworks* (STRC 2006). https://www.strc.ch/2006/Frejinger_Bierlaire_STRC_2006.pdf
- Frejinger, E., & Bierlaire, M. (2007). *Capturing correlation with subnetworks in route choice models*. Transportation Research Part B, 41(3), 363–378.
- Fosgerau, M., Frejinger, E., & Karlstrom, A. (2013). *A link based network route choice model with unrestricted choice set* (Recursive Logit). Transportation Research Part B.
- Prato, C. G. (2009). *Route choice modeling: Past, present and future research directions*. Transport Reviews.（综述，含 overlap/choice set 讨论）

### Trajectory generation / multi-modal evaluation
- Salzmann, T., Ivanovic, B., Chakravarty, P., & Pavone, M. (2020). *Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data* (ECCV 2020).（ADE/FDE、KDE-NLL、Best-of-N）
- Cui, H., Radosavljevic, V., Chou, F.-C., et al. (2018/2019). *Multimodal Trajectory Predictions for Autonomous Driving using Deep Convolutional Networks* (arXiv:1809.10732).（MTP loss / 多模态预测）
- Yuan, Y., & Kitani, K. (2020). *Diversifying Trajectory Forecasting with ...*（提出/使用 APD、multi-modal metrics 的一系列工作；在后续论文如 Motron 中被沿用）。
- Salzmann, T., et al. (2022). *Motron: Multimodal Probabilistic Human Motion Forecasting* (CVPR 2022).（APD + multi-modal ADE/FDE）

### Corridor extraction / trajectory clustering / network path clustering
- Zygouras, E., & Psyllidis, A. (2017). *Defining and finding corridors and routes from trajectories: A literature review*. International Journal of Geographical Information Science.
- Jiang, S., et al. (2015/2016). *Discovering Urban Bike Sharing Primary Corridors from Trajectories*（使用轨迹相似性/聚类抽取 corridors 的代表性 pipeline）。
- Wang, Z., et al. (2020). *k-paths: Representative Paths in Road Networks* (VLDB 2020).（path-level aggregation / representative path extraction）
- Hunter, T., Herring, R., Abbeel, P., & Bayen, A. (2009). *Path inference filter: A framework for probabilistic map matching, path inference, and travel time prediction from GPS data*.（从稀疏 GPS 得到路径分布，可用于 OD/corridor 份额估计）

---

## 6. （补充）Crowdsourced GPS 下 corridor diversity 的“ground truth”与解释

> 你指出的核心矛盾非常关键：**同一 OD 下出现多条轨迹**，在“行为解释”上到底代表什么？

### 6.1 Route choice 研究里，“corridor diversity”的 ground truth 通常从哪来？

在传统 route choice（离散选择）研究中，“真实选择 / ground truth”一般来自：

- **Revealed preference（RP）**：每次出行（trip）都有“最终选了哪条路”的观测（可来自 GPS、ETC、车载设备、浮动车等），从而可以把“被选择的路径”视为 ground truth。
- **Stated preference（SP）**：受访者被要求在若干候选路线中选择（或对属性做偏好陈述），因此每个选择实验有明确的 ground truth。
- **Simulation / assignment**：在可控环境中，ground truth 是模型/仿真生成的选择。

**关键点**：传统 route choice 里，ground truth 是“每个 trip 的 chosen route”。而所谓 corridor diversity，往往是对这些 trip 的选择结果在 OD 层面做聚合（统计分布）。

### 6.2 众包 GPS（WorldTrace）里：同一 OD 的多条轨迹，是“同一人多次选择”还是“多人各选一次”？

众包数据里往往**同时存在两类机制**：

1) **同一人（或同一设备/车辆）多次走同一 OD**
- 这时 corridor diversity 更多反映：
  - 个体决策的随机性（stochasticity）
  - 时变性（time-of-day / congestion / incident / learning）
  - 计划与执行误差（导航提示、临时绕行）

2) **不同人（不同设备/车辆）在同一 OD 上选择不同路线**
- 这时 corridor diversity 更多反映：
  - 偏好异质性（heterogeneity）：收费偏好、时间价值 VOT、熟悉度、风险规避、驾驶风格
  - 人群结构差异：本地 vs 外地、职业司机 vs 私家车

因此在**没有“个体 ID”的情况下**，你当前从 GPS 轨迹直接算出来的 corridor diversity，本质上是一个**混合量**：

> corridor diversity =（人群异质性）+（个体时变/随机性）+（观测噪声与 OD 聚合误差）

这不是“不可用”，但你必须在论文里把它的解释口径说清楚。

### 6.3 WorldTrace 的 `Owner` 字段能否区分个体/人群？

WorldTrace 的元数据（Meta.zip）中包含 `Owner` 字段，用于标识轨迹上传者（uploader）。这意味着你至少拥有一个**可用于 panel 分析的伪 ID**（proxy identifier），从而在方法上可以做“同一 owner 内 vs owner 间”的分解。  

- WorldTrace 数据集概况、覆盖范围与预处理字段（含 map matching 的 `osm_way_id` 等）见其数据卡说明。  

> 注意：`Owner` = uploader 并不必然等价于“唯一自然人”。它更像：账号/设备/车队/组织的标识。

**建议的处理方式（方法论上可自洽）：**

- 把 `Owner` 当作**最小粒度的可重复观测单元**（panel entity）。
- 用数据做 sanity check：
  - 每个 Owner 平均有多少条轨迹？轨迹是否跨多国/多城市（可能是组织或共享账号）？
  - Owner–OD 对的重复次数是否足够支撑“同一人多次选择”的推断？

### 6.4 文献里如何处理“是否同一人”的问题？（可直接借鉴的两条路线）

#### 路线 A：有个体/车辆 ID → 做 panel / within-person 分析
- 有些 GPS 研究明确利用“司机的长期轨迹历史”构造 route choice map（本质上允许同一人多次出行，因此可以谈个体的路线选择分布与稳定性/变化）。

#### 路线 B：没有个体 ID → 把每条轨迹视作独立样本（population-level）
- 很多基于大规模 GPS 的 choice set / route identification 工作，本质是从观测到的 route 分布中抽取“代表性路径/候选集”。这类工作往往不需要（也不假设）能够把同一 OD 的多条轨迹归因到同一人；它们对 diversity 的解释通常是“人群层面的路线分布”。
  - 例如 DDPI（data-driven path identification）直接从观测轨迹中抽取 OD 的“唯一观测路线”作为 choice set（隐含口径：**不同路线的存在=人群层面的多样性**）。
  - 例如在卡车 GPS 流数据上评估 BFS-LE 生成的 choice sets 时，观测路线来自“truck GPS traces stream”，同样是把轨迹当作样本集合来评估覆盖与冗余。

### 6.5 你可以在论文里写得更“硬”的：把 corridor diversity 做成可分解指标

为了把“同一人 vs 不同人”问题落地为可量化结论，建议你把 diversity 报告为两层：

1) **Within-owner diversity（个体/设备内）**
- 对每个 `(Owner, OD-zone)`，把该 owner 在该 OD 的轨迹聚类成 corridors，并计算熵/有效 corridor 数。
- 解释：个体随机性 / 时变性。

2) **Between-owner diversity（owner 间异质性）**
- 对固定 `OD-zone`，比较不同 owner 的 corridor 分布差异（比如用 JSD / KL / Earth mover’s distance）。
- 解释：偏好异质性。

3) **Total diversity（总体）**
- 把所有轨迹混在一起算的 diversity，作为“总体观测多样性”。

（如果你用 Shannon entropy 作为 corridor 分布的 diversity，可以进一步用经典的 entropy decomposition：总熵 = 组间熵 + 加权组内熵。）

---

## 7. （补充）OD-bin 粒度选择：交通 vs ML 的典型做法与对 corridor diversity 的影响

### 7.1 交通领域：TAZ / census tract 等 zone-based OD

交通建模通常不会用“精确 OD 坐标匹配”，而是：

- **TAZ（Traffic Analysis Zone）**
  - 常见经验：TAZ 规模在城市核心区更细（可到街区/blocks 级），郊区更粗。
  - 早期观点常把 TAZ 设计得接近 census block group 的粒度（block group 典型人口量级 600–3000），并指出 CBD 区域需要更小的 zone。  

- **census tract / block group**（或其变体）
  - tract 往往更粗（平均人口更大）。

**一个可引用的“量级对照”**（以 Baltimore 区域 TAZ 为例）：
- 公开 TAZ 数据描述里给出了 **平均 zone 面积约 350 acres**（约 0.55 平方英里）。
  - 换算成等面积正方形边长约 1.19 km（因此你 `od_bin_deg=0.01`≈1 km 在量级上并不离谱）。

### 7.2 ML（trajectory generation / prediction）领域：常见是 grid cell / exact matching

- **Trajectory prediction（如自动驾驶/行人预测）**：往往条件是“同一场景/同一 agent 的历史”，OD 是隐含的，不会做 zone-based OD 聚合。
- **Urban mobility trajectory generation（如 GTG / Cardiff）**：常见做法是用 grid（栅格）统计空间分布（例如 Cardiff 用 30×30 grid 来计算 JSD-SD；也用 OD grid 计算 JSD-trip）。

### 7.3 粒度对 corridor diversity 的结构性影响

OD zone 粒度越粗，你会越频繁地把“真实不同 OD 的轨迹”混到一起：
- corridor 数会增加（多出的是“不同真实 OD 对应的不同主干路”）
- corridor 之间距离会拉大（更像“不同旅行目的”而非“同一 OD 下的路线替代”）

OD zone 粒度越细：
- corridor 更接近“路线替代”（route alternatives）
- 但样本会稀疏（每个 OD 内轨迹数变少），corridor diversity 的估计方差变大

因此文献与实践里常见的做法是：
- **先选一个交通解释上合理的 zone（TAZ/tract/grid）**，再做 sensitivity analysis（至少 2–3 个粒度）。
- 你当前的 `od_bin_deg=0.01` 可以作为“中等粒度”基线，但最好补上：0.005 / 0.02（或更贴近 TAZ/tract 的替代）做敏感性。

另外，交通分配/建模领域有长期文献讨论 zone size 对结果的影响（例如不同平均宽度/面积的 zoning system 会改变 assignment 结果），这为你做粒度敏感性提供了“合理性背书”。

---

## 8. （补充）同一 OD corridor 数量的上界：choice set generation 的经验法则

### 8.1 为什么必须“截断”？
理论上同一 OD 的 feasible paths 可能无穷多（尤其在连续路网、允许绕行、允许回路时）。
在 route choice 中，这会直接导致：
- choice set 不可枚举
- overlap/correlation 的处理必须依赖有限候选集

因此 route choice 文献几乎都会显式选择：
- **choice set generation**（k-shortest paths, link elimination/penalty, labeling, simulation sampling, branch-and-bound 等）

### 8.2 k-shortest paths 的 K：文献里常用多少？
没有统一标准，但有相对稳定的量级：

- 综述中提到 Bekhor 等人的实验里使用过 **K=15** 与 **K=40** 的 k-shortest path sets，并讨论 choice set size 对模型与结果的影响。  
- 也有实证工作将 k-shortest path 的上限设到 **60**（如 Simonelli 等人的 choice set generation 比较中设置 maximum k=60）。

更近年的 choice set generation 基准评估也强调：算法效果对输入参数（包括 K）敏感，需要做定量比较与参数选择，而不是固定拍脑袋。 

### 8.3 “K 太小/太大”的讨论口径（你可以直接写进方法章节）

- **K 太小**：
  - 漏掉“真实存在但不是最短”的 corridor（尤其是收费/规避拥堵/偏好主干路等原因）
  - 造成 corridor diversity 的系统性低估

- **K 太大**：
  - 引入大量极不可能、甚至行为上不合理的绕行路径
  - overlap 结构会被噪声稀释，导致 corridor clustering / overlap penalty 失真
  - 计算成本上升

**实践经验（建议写成“经验法则 + 实证校准”）：**
- 用“观测轨迹覆盖率（coverage of observed routes）”或“authenticity/redundancy”作为 K 的选取准则。
- 如果你采用“观测轨迹→corridor”的数据驱动 approach（而不是枚举候选路径），K 的问题转化为：
  - 你对 corridor 聚类后的 **最小 cluster size / 最小频率阈值** 选多少；
  - 以及“长尾稀有路线”是否被保留。

---

## 9. （补充）2024–2025 轨迹生成工作里 corridor / multimodality 的定义与评估

你提到的 Cardiff / DiffPath / GTG 这类工作，严格说更接近“**mobility trajectory synthesis / road-path generation**”，而不是自动驾驶里的“motion forecasting”。所以它们的“多模态/多样性”口径与 Trajectron++ 那套 minFDE/APD 并不一样。

### 9.1 Cardiff（2025）：分层生成，但评估主要是“分布一致性”而非 corridor diversity

- **Corridor / mode 定义：隐式**
  - Cardiff 在模型结构上显式引入“segment-level（道路段序列）→ GPS-level（细粒度点）”的层级，但它并不显式做 corridor clustering。

- **评估指标：以 Jensen–Shannon Divergence 为主的 dataset-level realism**
  - Cardiff 采样 2000 条合成轨迹，并基于 JSD 定义三类指标：
    - **JSD-SD**：空间分布（把城市划成 30×30 grid，统计落点分布）
    - **JSD-LD**：轨迹长度分布
    - **JSD-trip**：OD grid 分布

> 这组指标可以看作“宏观分布层面的 diversity/coverage”，但它不会直接回答“同一 OD 下有多少条不同 corridor”。

**对你工作的启示：**
- 你可以把 Cardiff 的 JSD-SD / JSD-trip 当作你 corridor diversity 之外的“宏观 sanity check”：
  - corridor 多样性高但宏观分布偏离大 → 可能是噪声/OD 聚合过粗

### 9.2 GTG（2025）：宏观（JSD）+ 微观（序列距离），仍然不是显式 corridor

GTG 的目标是跨城市生成 road-segment trajectories。

- **评估指标**分两类：
  - **Macro metrics**：用 **JSD** 比较真实 vs 生成数据在三种统计特征上的分布：
    - Distance（旅行距离）
    - Radius（radius of gyration）
    - LocFreq（道路段访问频率）
  - **Micro metrics**：对每条生成轨迹与其对应真实轨迹的序列相似性，使用：
    - Hausdorff, DTW, EDT, EDR

> 同样，这套指标更像“distribution similarity + per-trajectory similarity”，并不显式输出 corridor 数量或 corridor 间分离度。

### 9.3 DiffPath（2024/2025）：显式在“路段访问分布”上做 divergence

DiffPath 是 road-network based path generation（latent diffusion）。它的评估更接近你要的 corridor / path distribution：

- **Similarity Score（SS）**：每条生成路径与最相似真实路径之间的重叠度（edge overlap / matching）。
- **KLEV / JSEV**：基于“edge visit frequency distribution”的 KL divergence / Jensen–Shannon divergence，用于评估生成路径的整体分布是否接近真实分布。

> 这本质上把“corridor”当作 **edge-usage distribution 的模式**（隐式），并用 divergence 评估多样性与逼真度。

---

## 10. 交给 partner 的需求清单（按优先级）

### P0（必须搞清楚才能继续）

#### P0.1 WorldTrace / 众包 GPS 数据里的 corridor diversity 如何解释？
- 把 corridor diversity 明确写成：
  - population-level heterogeneity
  - within-individual time variability
  - 以及 OD 聚合与 map matching 噪声的混合
- 用 `Owner` 做 panel 分解（within vs between），并验证 Owner 的可用性（重复度、空间跨度等）。

#### P0.2 找 1–2 篇“众包 GPS（非调查数据）”研究 route choice diversity 的文献
- 目标是找到**明确处理“是否同一人”的策略**：
  - 有 ID：panel（driver/vehicle/user）
  - 无 ID：population-level（独立样本）
- 推荐检索关键词：
  - "route choice" + "GPS traces" + "individual" / "driver" / "panel"
  - "floating car data" + "route choice" + "variability"
  - "Strava" + "route choice" / "cycling" + "GPS"

#### P0.3 OD zone 粒度的选择：交通领域典型尺度 + 敏感性分析证据
- 交通领域：TAZ/tract/block group 的典型尺度（最好能给出“典型直径/面积范围”）
- 找到至少 1 篇明确讨论“zone size 对结果敏感”的交通 assignment / OD estimation / route choice 文献

### P1（建议补充）

#### P1.1 Cardiff / DiffPath / GTG 的 corridor/multimodality 定义总结
- 它们是否：
  - 只做 minFDE/BoN？（大概率不是）
  - 或者使用 distributional metrics（JSD/KL）？
  - 是否引入显式 clustering/mode count？

#### P1.2 choice set size（K）经验法则
- 汇总不同论文中 K 的常见取值与调参方法
- 找到 1–2 篇专门讨论 choice set size trade-off 的论文（K 太小漏 corridor，K 太大引噪声）

### P2（锦上添花）

#### P2.1 LCS vs Longest Common Substring（连续）的比较
- 是否有文献比较两者在 corridor clustering 的表现？
- 若没有，至少找 1–2 篇使用 LCS/LCSS/substring 做轨迹相似性/公共子轨迹抽取的论文，为你写“方法选择动机”提供引用。

#### P2.2 Path Size Logit 的后续变体
- 综述类文章（route choice modeling review）中通常会覆盖 PSL、C-logit、error-components、以及其他 overlap correction 变体；需要确认哪些在近年更常用。

---

## 参考文献（本次补充涉及）

- WorldTrace 数据卡（字段、覆盖范围、时间跨度等）。
- WorldTrace 论文（KDD 2025, 数据集概述）。
- Cardiff（2025）：JSD-SD/JSD-LD/JSD-trip 指标与 30×30 grid 的定义。
- GTG（2025）：JSD（Distance/Radius/LocFreq）+ Hausdorff/DTW/EDT/EDR。
- DiffPath（2024/2025）：SS + KLEV/JSEV（edge visit frequency divergence）。
- Route choice 综述（Prato 2009）与 choice set generation 参数敏感性研究（Malhotra et al. 2024）。


### 参考链接（方便 partner / paper 直接引用）

- WorldTrace dataset card（HuggingFace）：https://huggingface.co/datasets/OpenTrace/WorldTrace
- WorldTrace paper（KDD 2025, OpenReview）：https://openreview.net/forum?id=CdN6vtOFP0
- Cardiff（arXiv 2507.13366）：https://arxiv.org/abs/2507.13366
- GTG（arXiv 2502.01107）：https://arxiv.org/abs/2502.01107
- DiffPath（OpenReview / ICLR 2025 withdrawn）：https://openreview.net/forum?id=1o3fKLQPRA
- Route choice modeling 综述：Prato, C. G. (2009). *Route choice modeling: Past, present and future research directions*. Journal of Choice Modelling.（EconStor 版）https://www.econstor.eu/handle/10419/66846
- Choice set generation benchmark：Malhotra, Advani, Bhaskar (2024). *Performance evaluation of path choice set generation algorithms for route choice modelling*. Journal of Intelligent Transportation Systems. DOI: 10.1080/15472450.2024.2373866
- TAZ 设计与尺度讨论（TRB 2017）：https://rosap.ntl.bts.gov/view/dot/57334
- Baltimore Region 2020 TAZ dataset（含平均面积等描述）：https://gisdata.baltometro.org/datasets/BMC::2020-traffic-analysis-zones-taz-for-baltimore-region/about
- GPS traces 路线偏好/个体历史示例：Duncan & Krumm (或相关版本) “Constructing route choice maps from GPS traces”（ResearchGate 入口）：https://www.researchgate.net/publication/220308453_Constructing_route_choice_maps_from_GPS_traces
- LCS/公共子轨迹抽取示例：Xie et al. (2016) “Detecting Road Intersections from GPS Traces Using Longest Common Subsequences”（ScienceDirect）：https://www.sciencedirect.com/science/article/pii/S092427161630047X
- Data-driven path identification（DDPI, observed unique routes as choice set）：Ton et al. (2018) *A data-driven approach for choice set generation in route choice modelling*（link to abstract / publisher page）：https://www.tandfonline.com/doi/abs/10.1080/21680566.2018.1430896
- BFS-LE（truck GPS traces, choice set generation evaluation）：Zhao et al. (2017) *Generating route choice sets for freight transport*（ResearchGate 入口）：https://www.researchgate.net/publication/312550597_Generating_route_choice_sets_for_freight_transport
- k-shortest paths 上限示例：Simonelli et al. (2020) *Comparison of route choice set generation algorithms...*（ResearchGate 入口）：https://www.researchgate.net/publication/343780252_Comparison_of_route_choice_set_generation_algorithms_for_freight_transport_demand_models_in_moderate_to_large_scale_transport_networks

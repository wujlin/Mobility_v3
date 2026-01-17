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

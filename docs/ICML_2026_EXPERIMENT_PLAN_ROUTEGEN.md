Implementation Plan, Task List and Thought in Chinese

# ICML 2026：Route Generation（CascadeTraj）实验思路与执行计划（PI Review Draft）

> 目标：用**可诊断的证据链**支撑论文主张——长程路线生成的核心瓶颈是“拓扑多模态导致的 mode interference/averaging”，而**决策-执行级联（先离散承诺，再连续执行）**能系统性缓解该失败模式；soft prior 提升可行性但不做 hard truncation；census 仅作为可选引导（extensibility），不把论文变成 population synthesis。

---

## 0) 范围与三条核心 Claim（写作/实验必须对齐）

### 0.1 本文不做什么（边界声明）
- **不做** urban rupture / avoidance field（这属于第二篇文章）。
- **不做** population synthesis 的完整闭环（个体属性生成不是本文主线）。
- census 相关只作为“可选语义引导/可控性扩展”的 **ablation**，用一段话在 Introduction 提前封堵歧义。

### 0.2 本文要证明什么（Claim → 需要的证据类型）
**Claim A（诊断发现）**：端到端连续坐标生成在 full-trip route generation 上会出现模式坍塌，其根因是 corridor-level 多模态在连续输出空间发生 mode interference，导致 destructive averaging（“直线/模糊走廊”）。
- 证据：**Fig.1/2 级别的可视化诊断** + dataset-level 的 diversity/coverage 指标崩溃（不是只报 FDE/ADE）。

**Claim B（机制性解法）**：用“决策（离散/稀疏承诺）→执行（连续生成）”的级联分解，可显式承载拓扑多模态，从机制上打破 averaging trap，恢复 corridor-level diversity，同时保持 intent consistency。
- 证据：对比端到端 baseline vs CascadeTraj，在**同一条件**下多样性显著提升，同时 intent/realism 不退化。

**Claim C（鲁棒性与可行性）**：地图信息更适合作为 **soft prior**（road proximity / dist-to-road / road_prob），而非训练期 hard mask/裁剪；hard truncation 能“看起来更可行”但会把分布上限绑定到 proxy 质量并削掉真实模态。
- 证据：soft prior vs hard mask 的 ablation：可行性提升同时多样性不被截断；并做 dilation/buffer sensitivity audit 避免被 mask 孔洞污染（见 `docs/README.md` 的踩坑提醒）。

---

## 1) 数据与路径（按仓库合同口径，避免“路径不一致导致无法复现”）

### 1.1 外置数据根目录（推荐）
按 `docs/DATA_STRUCTURE.md`：Phase D 默认外置 `$RAW_ROOT`（不进 git）。

典型结构（示例，实际以你机器为准）：
- `$RAW_ROOT/worldtrace/<city>_core_v1/segments.parquet`
- `$RAW_ROOT/worldtrace/<city>_core_v1/osm_road_prob.npy`（或 dist-to-road/road_mask）
- `$RAW_ROOT/worldtrace/<city>_core_v1/poi_density_*.npy` / `landuse_*.npy`
- `$RAW_ROOT/census/<city>_core_v1/...`（若启用 census ablation）

> 口径真相源：`docs/DATA_CONTRACT.md`（bbox/grid/坐标/road_prob 定义）。

### 1.2 立刻要跑通的“最小可训练样本形态”
为了支持 route generation（O,D,t0,context → full route），数据至少要能提供：
- 轨迹序列（位置序列）与时间戳
- trip-level 的 origin/destination
- 可选的 context channels（OSM soft prior，POI/imagery，census covariates）

如果你当前的 WorldTrace 产物还是 parquet/segments 形态，建议先做一个**轻量转换**（不追求最优 IO，只求 48 小时内跑通 baseline）：
- 从 `segments.parquet` 采样一个可控规模的子集（例如 50k–200k segments）
- 统一投影到 `(y,x)` 栅格（bbox/grid 以合同写死）
- 导出一个“训练可直接 mmap/顺序读取”的格式（parquet 分区/arrow/npz 均可，KISS 优先）

> 避免一次性工程化：先跑通“诊断 baseline + 关键图”，再考虑 manifest/更高效格式（`docs/WORDTRACE_UNITRAJ.md` 的 IO 风险提醒）。

---

## 2) 任务定义（实验口径写死，避免审稿人觉得你在换任务）

### 2.1 条件与输出
条件（最小）：$c = (o, d, t_0)$  
可选条件：OSM soft prior、POI/imagery、（ablation）census covariates  
输出：完整路线 $\tau = (p_1,\dots,p_T)$ 或等价的 action/velocity 表示（最终都可还原到位置序列用于评估）。

### 2.2 采样设置
- 生成模型统一采样 `K` 条（建议 `K=20`，快速阶段可 `K=5`）。
- 所有模型统一随机种子集合（至少 3 个 seed），否则 diversity 对比不可比。

---

## 3) 模型与 Baseline 设计（只保留能回答 Claim 的最小集合）

### 3.1 必要 Baselines（直接对打）
1) **End-to-End AR**：自回归坐标/位移模型（检验 drift 与误差累积）。
2) **End-to-End Diffusion/Flow**：端到端连续序列生成（直接暴露 mode collapse/averaging）。
3) **Hard mask / hard support（诊断项）**：训练期或采样期的输出裁剪/masked softmax 版本（用于展示“可行性来自 truncation，并非真正学到分布”）。

> 备注：Hard mask baseline 的定位是“诊断/止损线”，不要把它写成主贡献能力（与仓库既有经验一致）。

### 3.2 我们的方法（CascadeTraj）
**Stage-1 决策（Topological Commitment）**  
- 输出：稀疏 waypoint skeleton（如 2 个 waypoint + end anchor）
- 模型：AR 或 diffusion 均可（以能稳定表达多模态为准）

**Stage-2 执行（Physical Execution）**  
- 输入：waypoints + (o,d,t0,context)
- 输出：连续路线（全分辨率）
- 重点：执行层的目标是“像真 + 可行”，但不应承担“创造拓扑模态”的责任（模态应在 Stage-1）。

### 3.3 分阶段 Go/No-Go（避免 10 天内陷入工程黑洞）
- **Gate-0（数据可用）**：能生成 GT 轨迹可视化 + 能采样若干 OD 的候选样本（哪怕是 dummy baseline）。
- **Gate-1（诊断成立）**：端到端 diffusion 在关键 OD 上出现平均化/走廊混叠（Fig.1 的左半边成立）。
- **Gate-2（级联有效）**：仅加入“离散承诺（waypoints）”就能让多走廊样本显著分离（Fig.1 右半边成立），且 diversity 指标显著上升。
- **Gate-3（soft prior 提升可行性）**：soft prior 提升 on-road proxy 且不过度削多样性；hard mask 的“提升”需伴随分布截断的证据。

---

## 4) 指标体系（必须覆盖“多模态”，否则 Claim A/B 站不住）

### 4.1 Intent consistency（长程一致性）
- 终点误差（FDE/endpoint error）
- 到达率/终止一致性（是否到达目的地邻域）

### 4.2 Realism（几何与形状）
- DTW / Fréchet 类形状距离（比单点 FDE 更能抓“形状像不像”）
- 路径长度/绕行比的分布一致性（避免“看似到达但走直线”）

### 4.3 Feasibility proxies（可行性：必须做敏感性审计）
- on-road ratio / off-road rate：基于 OSM road mask 或 dist-to-road
- **敏感性审计**：buffer/dilation/threshold 扫描，避免“mask 孔洞”把模型冤死或把 hard mask 美化（对应 `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md` 的经验）

### 4.4 Diversity / Coverage（核心卖点）
我们需要一个“能区分**一条好路线** vs **多条不同走廊的好路线**”的指标族。建议采用两层口径（KISS + 可复现）：

**(a) 条件内多样性（intra-condition diversity）**  
给定同一条件 $c=(o,d,t_0,ctx)$，对生成的 $K$ 条路线：
- **Pairwise Jaccard Distance（占用栅格集合）**：把路线 rasterize 到低分辨率网格（例如 64×64 或 128×128）得到占用集合 $S(\tau)$，定义
  - $D_{\text{Jacc}} = 1 - \frac{|S(\tau_i)\cap S(\tau_j)|}{|S(\tau_i)\cup S(\tau_j)|}$
  - 汇总：均值/分位数（越大表示越多样）。
- **Self-BLEU（离散 token 序列）**：将路线压缩为网格 token 序列（去重/下采样），计算 self-BLEU（越低越多样）。

**(b) 走廊覆盖（corridor coverage）**  
核心是“是否覆盖到 GT 的主要走廊模态”。最小可行做法：
- 在每个条件下，对 GT 路线做聚类得到走廊簇 $\{\mathcal{C}_m\}$（特征可用 occupancy grid 或 polyline embedding；聚类算法 DBSCAN/KMeans 均可，先用 KMeans 让流程跑通）。
- 生成样本覆盖率：生成集合与 GT 簇的匹配比例（例如每个 GT 簇是否至少被一个生成样本命中，或按簇权重算 recall）。

> 通过这两层指标，我们能把“端到端模型看似 FDE 还行但其实 collapse 到单走廊/平均走廊”的问题量化出来，从而支撑 Claim A/B。

---

## 5) 关键图与可视化交付物（按 `docs/visual_style_guide.md` 执行）

### 5.1 风格与工程规范（必须统一）
- 统一入口：`src/plot_style.py`（source-of-truth 在 `src/visualization/plot_style.py`）
- 配色：Okabe–Ito（`OKABE_ITO`），线宽/字号/figsize 按 `PaperStyle` 与 `FIGSIZE_HALF/FULL`
- 输出：主图 PDF（矢量），预览 PNG 可选
- **禁止** `bbox_inches="tight"`（避免 bbox 抖动导致 LaTeX 子图错位）

### 5.2 Fig 1 / Fig 2 级别“必出图”（支撑 Claim）
**Fig 1：Mode collapse 诊断图（主文级）**  
同一组 $(o,d,t_0)$ 条件下：
- 左：End-to-End baseline 的 $K$ 样本（平均化/直线/走廊混叠）
- 右：CascadeTraj 的 $K$ 样本（多走廊清晰分离）
- 视觉编码建议：
  - GT：黑色粗线（或灰色）
  - 样本：蓝/绿系 + alpha（多条叠加）
  - 可加 waypoint 标记（点/十字）强调“离散承诺”确实在分离模态

**Fig 2：Diversity–Realism Tradeoff（主文级）**  
散点或 Pareto 曲线：x=realism（DTW/Fréchet），y=diversity（Jaccard/self-BLEU/coverage），把“多样性不是靠牺牲真实性换来的”讲清楚。

> 其余 ablation 图（soft prior / hard mask / semantics / census）放 Fig 3/4 或 SI。

---

## 6) 实验矩阵（最小闭环优先；每项都回答一个 Claim）

### E0（必做）数据与评估管线自检
目标：确保每个指标/图都能在小样本上跑通，避免最后两天才发现口径问题。
通过标准：能生成 Fig1 的“同条件多样本叠图”；能输出四类指标（intent/realism/feasibility/diversity）。

### E1（必做）端到端 baseline 诊断（支撑 Claim A）
目标：证明 collapse 真实发生，且是拓扑多模态造成的。
输出：Fig1 左半边 + diversity/coverage 指标显著偏低。

### E2（必做）CascadeTraj（仅 Stage-1 改动）验证（支撑 Claim B）
目标：只靠“离散承诺”就能让走廊模态分离（避免把功劳归因给执行层）。
输出：Fig1 右半边 + diversity/coverage 显著提升；intent 不显著变差。

### E3（中）执行层细化（支撑“physical execution”叙事）
目标：在已分离走廊模态的前提下，提升 realism/feasibility（形状/速度纹理/局部几何）。
输出：realism 指标提升，同时 diversity 不回落。

### E4（中）soft prior vs hard mask（支撑 Claim C）
目标：展示 hard mask 的“可行性提升”伴随分布截断/多样性损失；soft prior 更平衡。
输出：feasibility↑ 且 diversity 不被截断；并提供 dilation/buffer 敏感性审计（防止 proxy 污染）。

### E5（可选）语义通道与 census guidance（不抢主线）
目标：展示可控性/扩展性，而不是把论文变成 population synthesis。
输出：只在补充实验/ablation 中呈现；主文只保留边界声明与一句话结果摘要。

---

## 7) 结果落盘与可复现约定（避免“跑完找不到产物”）

建议统一外置实验根目录（示例）：
- `$RAW_ROOT/experiments/icml2026_routegen/<exp_name>/`

每个实验目录最少包含：
- `config.json`：所有超参/数据口径/seed
- `metrics.json`：四类指标 + 置信区间/多 seed 汇总
- `samples/`：用于 Fig1 的同条件多样本可视化（PDF+PNG）
- `audit/`：feasibility proxy 的敏感性审计结果（不同 dilation/buffer 的曲线/表）

日志习惯（来自 `docs/README.md` 踩坑）：优先 `python -u ... |& tee logs/xxx.log`，不要后台重定向导致“看似没进度”。

---

## 8) 风险与止损线（10 天窗口下必须严格止损）

- **数据/指标先于模型**：48 小时内必须跑通 E0/E1 的 Fig1 左半边，否则先停模型开发。
- **避免 hard support 变成主路线**：hard mask 只作为诊断对照，结果呈现必须带“截断证据”与敏感性审计。
- **多进程/锁风险**：如遇 HDF5/多进程卡死，优先 `HDF5_USE_FILE_LOCKING=FALSE` 或 `--num_workers 0`（见 `docs/README.md`）。

---

## 9) 10 天时间盒（建议节奏；可按实际压缩）

- Day 1–2：E0（数据/评估/作图管线跑通）+ Fig1 样例框架定稿（遵循 `docs/visual_style_guide.md`）
- Day 3–4：E1（端到端 baseline collapse 证据 + diversity 指标）
- Day 5–6：E2（CascadeTraj 的 Stage-1 版本，优先把 Fig1 右半边做“扎实”）
- Day 7–8：E3/E4（执行层提升 realism + soft prior/hard mask 对照 + proxy 敏感性审计）
- Day 9：E5（可选语义/census ablation）+ 主文图表清理
- Day 10：复现实验/重跑关键 seed + 论文整合与查漏补缺

Implementation Plan, Task List and Thought in Chinese：本文档服务于 Phase D（WorldTrace×Detroit）的 essay 写作，强调“科学叙事”而不是“技术报告”。

# Essay 写作指南（Phase D：Behavioral Reference Frame → Behavioral Avoidance Field）

> 目标：把项目从“做了什么”写成“发现了什么、为什么重要、证据是什么”。  
> 单一真相源：数据口径以 `docs/DATA_CONTRACT.md` 为准；任务/评估口径以 `docs/TASK_DEFINITION.md` 为准；关键概念命名以 `docs/RESEARCH_LOG.md` 为准。  
> LaTeX 主稿：`essay/main.tex`（已按该叙事框架组织）。

---

## 0) 一句话主线（写在 Abstract 末尾、Introduction 末尾都成立）

我们用跨城市迁移构造一个 **Behavioral Reference Frame（行为参照系）**，并把“参照系预测的正常路线 footprint”与 Detroit 的真实路线 footprint 做差，得到一个具有**空间具体性**的 **Behavioral Avoidance Field（行为回避场）**，用来描述“断裂长什么样”（走廊替代、边界效应、断裂带形态），而不仅是“断裂强不强”（一个相关系数）。

---

## 1) 两个概念怎么用（避免叙事混乱）

- **Behavioral Reference Frame**：方法论概念（你构造 baseline 的方式）。标题/摘要/引言重点讲它。
- **Behavioral Avoidance Field**：实证发现/结果概念（你在 Detroit 上得到的空间残差场）。结果/讨论重点讲它。

叙事约束：不要把“destination gravity”等技术细节当主线；它只解释“为什么要用决策-执行分解/AR，否则参照系会退化成最短路近似”。

---

## 2) 写作结构（每节只回答一个问题）

### Abstract（回答：我们发现了什么？为什么可信？）
- 现象：功能断裂是空间异质的，标量指标难以描述其结构。
- 方法：行为参照系（从 functional 城市学到“正常 route choice”）。
- 产物：回避场（空间残差），可以揭示走廊替代与行为边界。
- 可信度：多源对齐（WorldTrace + OSM soft prior + POI + 遥感 + census）与可审计的数据契约。

### Introduction（回答：为什么 route choice 是“独特信号”？）
建议写成 5 段：
1) 城市功能断裂的重要性（Detroit 的空间异质性是动机，不是背景噪音）。
2) 传统指标是 outcome，不是 process（它们很重要，但缺少“人如何适应”的过程信号）。
3) 移动行为是“行为投票”，但 detour 很常见；最短路不是好的行为基线。
4) 行为参照系：用 functional 城市学习正常权衡；避免把“正常绕行”误判为“问题回避”。
5) 空间具体性：标量比较（detour ratio/相关系数）解释力有限；我们要回答“断裂结构是什么样”。

### Materials \& Data（回答：我们依赖哪些信息？各自提供什么“语义”？）
按“功能”写，不要按“下载步骤”写：
- WorldTrace：提供 OD+时间语境下的 route choice 事实样本（并明确 1Hz、matched 坐标与质量闸门）。
- OSM（soft prior）：提供“更像路”的连续概率场（road\_prob），作为特征与软正则，而非硬裁剪。
- SafeGraph POI：提供功能语义（区域“是什么/有什么”），解释非几何原因的绕行。
- Wayback 遥感：提供建成环境外观（POI 覆盖不到的结构：水体/绿地/大型设施）。
- Census/ACS：提供独立外部指标（vacancy/income/pop），用于验证回避场是否与断裂 proxy 对齐。

### Methodology（回答：如何把“参照→残差→空间场”做成可证伪的科学测量？）
按研究设计写成四步（与 `essay/sections/03_methodology.tex` 对齐）：
1) 在 functional 城市学习“正常 route choice”（构造参照系）。
2) 不在 Detroit 上重训（否则参照系不再是 counterfactual）。
3) 用一致的 OD+时间语境生成“正常 footprint”。
4) 与 Detroit 真实 footprint 做差并空间化：得到回避场 + 替代场（substitution）。

同时明确三条假设（H1/H2/H3），并说明每条假设对应的可证伪证据。

---

## 3) 图表怎么选（以“空间结构”说服读者）

建议把图表资源集中在“空间具体性”上：
- Fig 1（概念图）：Behavioral Reference Frame → Behavioral Avoidance Field（流程示意）。
- Fig 2（标量不足的证据）：Detroit vs Columbus 的 detour 标量相近（或弱相关），引出“需要空间场”。
- Fig 3（核心结果）：Detroit 的 avoidance field 地图（under-traversal）+ substitution field（over-traversal）。
- Fig 4（外部验证）：回避场聚合到 tract 后与 vacancy/income 的关系；同时给出一张对齐地图（不只给相关系数）。

> 注意：如果最终回避场只是 vacancy map 的低分辨率版本，必须展示“额外信息量”（走廊替代、边界更尖锐、时变结构等）来回答“为什么值得”。

---

## 4) 结果还没跑完时怎么写（不造假也不失分）

- Results 部分可以先写成“分析计划 + 预注册式口径”（用 will/report/define），并把“将报告的指标与可视化”写清楚。
- 任何数值型结论（相关系数、显著性、热点位置）必须来自仓库内的真实产物；否则只写“我们将用 X 验证 Y”。

---

## 5) 写作自检（避免写成工程日志）

每写完一节，问自己三个问题：
1) 这节回答的科学问题是什么？（一句话能说出来）
2) 读者读完会记住一个“发现/洞见”吗？还是只记住“我们做了很多步骤”？  
3) 如果删掉所有“我们接下来/首先/然后”，逻辑还能自洽吗？


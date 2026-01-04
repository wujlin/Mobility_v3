Implementation Plan, Task List and Thought in Chinese

# Research Log（实时记录｜口径与叙事统一）

> 目标：把“关键决策、口径、争议点、已证伪/未证伪的结论”集中到一份可追溯记录里，避免团队在不同文档里互相覆盖。  
> 原则：只记录能影响结论的变化（数据口径 / 任务口径 / 评估口径 / 叙事口径 / 止损线），不写执行流水账。

---

## 0) 两个核心概念（命名已拍板）

- **Behavioral Reference Frame（行为参照系）**：从 *functional reference cities* 学到的“正常 route choice 规律”的 baseline，用于回答“在相同 OD+时间语境下，正常情况下会怎么走”。它是**方法论概念**（baseline 的构造方式）。
- **Behavioral Avoidance Field（行为回避场）**：将 Detroit 上“预测的正常行为”与“真实行为”之差做空间化后的残差场，用于回答“哪些区域/走廊被系统性绕开”。它是**结果/发现概念**（residual 的空间结构）。

叙事约束：
- Title/Abstract/Introduction 重点强调 **Behavioral Reference Frame**。
- Results/Discussion 再引入并解释 **Behavioral Avoidance Field**（残差场的空间结构与外部指标对齐）。

---

## 1) 主线问题与科学假设（当前论文 framing）

科学问题：
- 城市功能断裂如何在人的移动（尤其是 route choice）中留下可检测的 signature？
- 用“行为参照系”（而不是最短路）能否把“正常绕行”与“断裂导致的回避”分离出来？

假设（对应 essay/sections/01c_research_question.tex）：
- **H1**：functional→functional 迁移残差应小（正常行为规律具有可迁移性）。
- **H2**：Detroit 的残差应形成空间上 coherent 的场，并与独立断裂指标相关。
- **H3**：行为参照系相比最短路参照系，能更好地隔离“正常 detour”，使 residual 更 localized、更可解释。

---

## 2) 数据与口径（Phase D 主线｜与合同绑定）

单一真相源：
- `docs/DATA_CONTRACT.md`（版本/坐标系/bbox/grid/软先验定义）
- `docs/TASK_DEFINITION.md`（任务定义/无泄漏/评估协议）

关键拍板（摘要）：
- 主数据：WorldTrace（1Hz；WGS84；优先使用 matched 坐标，质量闸门见合同）
- Detroit core：bbox `[-83.25, 42.25, -82.95, 42.50]`，grid `1024×1024`
- OSM：**只做输入特征/软先验/审计 proxy**（不做 hard cut）
- SafeGraph POI：统一到粗粒度类别并做栅格化（时间戳与有效性规则写入合同）
- Wayback 遥感：按 release_id 落盘（避免 release_date 缺失/异常造成口径漂移），下载与代理/SSL 口径见 `docs/WAYBACK.md`

---

## 3) 为什么从 Hard Support 转向 Soft Prior（已证伪/已锁定的点）

结论（在 legacy pilot 中被反复触发的风险）：
- Hard Support（训练期 masked softmax / hard cut）会把模型学习目标变成条件分布 `p(x | feasible)`，并把上限绑定到外部 mask 质量；同时会放大“mask 孔洞/缺路”的系统性偏差。

因此 Phase D 主线的约束是：
- OSM road 信息进入模型的方式是 **soft prior/feature + 可审计消融**，而非输出空间裁剪。

注：legacy 深圳 dt30 的 hard support + AR + DetRes 结果与审计被归档为历史证据链，见 `docs/archive/legacy_shenzhen/PHASE_C_RESULTS.md`。

---

## 4) 路线生成的结构选择（当前共识）

核心分解：
- **Decision（宏观决策）**：route choice / corridor choice（主要不确定性）
- **Execution（微观执行）**：给定 corridor 的平滑执行（更接近确定性控制）

结构机制：
- **AR（autoregressive waypoint planning）保留**：作为“多步一致性”的结构保证（wp2 依赖 wp1，end 依赖 wp1+wp2）。
- **Diffusion 的定位**：只在“单模态 baseline 足够可信”之后，用于表达 *normal route-choice* 的多模态（多风格路线），不承担“可行性修复”责任。

Diffusion 触发门槛（当前口径，写入 `docs/DATA_CONTRACT.md` / `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`）：
- CUT（OSM proxy）< 5%
- corridor_error < 10%
- detour 方向性偏差（Δp50(dev/len)）< 0.1

---

## 5) 需要持续追踪的“口径威胁”（避免再次走偏）

- **时间一致性**：WorldTrace(2021–2023) vs POI/遥感/OSM 的版本月份；必须记录 vintage 并做敏感性审计。
- **proxy 依赖**：任何“road/feasible”的判断都必须说明是 OSM proxy 还是 data-driven proxy（轨迹密度），并版本化 buffer/dilation/sigma。
- **迁移叙事的自洽性**：functional→functional 残差若不小，则 Behavioral Reference Frame 的“可迁移性”假设不成立，需要先修 H1 再谈 Detroit。


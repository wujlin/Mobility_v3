Implementation Plan, Task List and Thought in Chinese

# Essay 写作指南（统一入口：避免拿错 paper）

> 目标：把项目从“做了什么”写成“发现了什么、为什么重要、证据是什么”，并确保 **文档—代码—数据**一致。  
> 单一真相源：数据口径以 `docs/DATA_CONTRACT.md` 为准；任务/评估口径以 `docs/TASK_DEFINITION.md` 为准；关键决策以 `docs/RESEARCH_LOG.md` 为准。  

本仓库当前有三份 LaTeX 稿件（不要混用叙事与实验口径）：
- **ICML 2026｜Route generation（当前主线）**：`essay_icml_cascadetraj/main.tex`
- **（备份/对照）routegen 同步稿**：`essay_population/main.tex`
- **Paper-2｜Rupture/Avoidance field（非 ICML routegen 主线）**：`essay/main.tex`

下文先给 ICML routegen 的写作主线；Paper-2 的写作规范放在文末“附录”。

---

## A) ICML 2026｜Route generation（`essay_icml_cascadetraj/`）

### A0) 一句话主线（写在 Abstract/Intro 末尾都成立）

route generation 的 corridor-level 多模态在连续坐标空间里会诱发均值塌缩/漂移；我们转向 **road-graph 上的结构化决策**：用少步数的 **waypoint AR** 产生走廊承诺，再用 **A\*** 连接保证路径合法性（可选再接 continuous execution/refinement）。

---

### A1) 叙事边界（新 PI/审稿人最容易误解的点）

- **不使用 window-level（F=256 滑窗）作为 route generation 证据链**：window 会把任务降级为短距离轨迹延续，走廊选择不存在（已用 `E_D0/E_W0` 统计审计证实）。
- **不把 rupture/avoidance field 写进 ICML routegen**：那是 Paper-2（`essay/`），只会造成 scope creep。
- **语义/POI/census 不是主线前提**：ICML routegen 的核心是“结构化走廊决策机制”；语义只作为“context 是否 informative”的 gate/扩展点。

---

### A2) 写作结构（每节回答一个问题）

建议把论文结构固定为“问题→诊断→机制→验证→局限”：

**Abstract**
- Problem：corridor-level multi-modality 使端到端连续生成失败（平均化/漂移）。
- Finding：走廊选择必须结构化（graph/waypoint commitment），否则无法稳定覆盖。
- Method：Decision（waypoint AR）+ Execution（A* 合法连接；可选 continuous）。
- Evidence：候选覆盖 gate（K-shortest 覆盖不足）+ 语义信息量 gate（time+tier AUC）+ waypoint AR 原型（success/bestJ）。

**Introduction**
- 把“route generation”明确为 segment-level（整段行程），不是短窗预测。
- 解释为什么“候选集分类（K-shortest）”在覆盖率上会失败（需要 gate 证据）。
- 亮出核心机制：用 waypoint-level 的少步 AR 承载 corridor commitment。

**Data**
- WorldTrace 的 city 子集（Detroit/Columbus），segment-level 过滤标准（chord/detour/min_points）。
- OSM → road graph（节点/边/tier），GT route → graph path（node sequence）。

**Method**
- Decision：waypoint AR（bin-classification）+ 可解释的 time+tier conditioning。
- Execution：A* 连接（合法性保证）；可选 continuous executor（扩展点）。

**Experiments/Results**
- Gate-1（候选覆盖）：说明为什么候选集不够。
- Gate-2（语义信息量）：`AUC(time+tier)` 是否 >0.6（支持 context-conditioned diversity 的前提）。
- T4 原型：waypoint AR 的 val acc、A* success/bestJ、oracle 上界；用可视化解释瓶颈来自“bin 粒度/实例化策略导致的走廊混淆”，而不是“600-step node AR 的累积误差”。

---

### A3) 图表优先级（避免“只堆指标”）

routegen 主线的图表建议：
- **Fig 1（Hero）**：GT vs L2（平均化）vs E2E（漂移/不可达）vs 我们（graph commitment）。
- **Fig 2（Gate）**：K-shortest 覆盖不足（bestJ 分布）+ 诊断可视化（GT 在路上但候选走别处）。
- **Fig 3（Go/No-Go）**：语义信息量 gate（AUC by feature：time / tier / time+tier）。
- **Fig 4（T4 结果）**：waypoint AR + A* 的 success/bestJ 分位数 + 典型失败可视化（解释“可达性约束”的必要性）。
  - 如果 success 已经稳定（例如跨城 disjoint-union 修复后），Fig 4 更应该强调 oracle ceiling 与 `wp_bin` 粒度的 trade-off，并用可视化展示“bin 内多走廊混淆”的失败模式。

---

### A4) 结果未收敛时怎么写（不造假也不误导 PI）

- 论文里所有数字都必须能点到仓库产物路径（`_sync/wsa/.../report.json` 或 `$RAW_ROOT/.../report.json`）。
- 如果某一阶段仍是 prototype（例如 oracle ceiling 偏低、需要更细 `wp_bin`），要把它写成“发现/诊断”而不是“性能结果”。  

---

## B) 附录：Paper-2 rupture/avoidance（`essay/`）

> 这条线与 ICML routegen 不应混用证据链；仅保留旧写作主线供需要时参考。

一句话主线（paper-2）：
- Behavioral Reference Frame → Behavioral Avoidance Field（空间残差场），用于描述 rupture 的“结构”而不仅是“强度”。

写作入口：`essay/main.tex`；原指南内容已迁移到该稿件结构中，避免继续扩散到 routegen 叙事。

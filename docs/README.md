# docs 索引（单一真相源）

> 目的：把“规范 / 结果 / 诊断 / 方案 / 写作素材”分层，避免文档互相打架，避免重复踩坑。  
> 原则：**`docs/TASK_DEFINITION.md` + `docs/DATA_CONTRACT.md` 是协议真相源**；其余文档要么是结果事实总结，要么是阶段性诊断与方案备忘。

---

## 1) 看哪份？（按需求）

- **我要确认任务定义/无泄漏/评估协议**：`docs/TASK_DEFINITION.md`
- **我要确认多源数据版本/坐标系/时间一致性（数据契约）**：`docs/DATA_CONTRACT.md`
- **我要跟踪关键决策与口径变化（实时记录）**：`docs/RESEARCH_LOG.md`
- **我要快速理解代码框架与模块关系**：`docs/CODE_STRUCTURE.md`
- **我要理解数据产物与字段**：`docs/DATA_STRUCTURE.md`
- **我要了解学术 SOTA（Map-based vs Hierarchical）与路线选择**：`docs/SOTA_TRAJECTORY_GENERATION_2025_UPDATE.md`
- **我要看新的路线图（OSM 道路先验（软） + 拓扑 + 城市语义 + AR + Diffusion 多模态）**：`docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`
- **我要把主数据切换到 WorldTrace（UniTraj 底座）并做 Detroit 试点**：`docs/WORDTRACE_UNITRAJ.md`
- **我要准备外部验证指标（ACS/TIGER tract-level vacancy/income/pop）**：`docs/DATA_CONTRACT.md`
- **我要确认工作站/本地运行口径（RAW_ROOT / 代理 / tmux / 多进程）**：`docs/WORKSTATION_GUIDE.md`
- **（Legacy｜封存）深圳 dt30 复现材料入口（不参与当前 Detroit story）**：`legacy/shenzhen/README.md`

### （归档）Phase B / 窗口级材料（保留但不作为当前主线入口）

- `legacy/shenzhen/docs/phase_b/PHASE_B_RESULTS.md`
- `legacy/shenzhen/docs/phase_b/PHASE_B_CFG_VISUALIZATION.md`
- `legacy/shenzhen/docs/phase_b/ROOT_CAUSE_ANALYSIS.md`
- `legacy/shenzhen/docs/phase_b/RESIDUAL_DIFFUSION.md`
- `legacy/shenzhen/docs/phase_b/SHRINKAGE_LITERATURE_ROADMAP.md`
- `legacy/shenzhen/docs/phase_b/PHASE_B_REVIEW.md`
- `legacy/shenzhen/docs/phase_b/RF_PILOT.md`
- `legacy/shenzhen/docs/memos/EXPERT_CONSULTATION_PACKET.md`
- `legacy/shenzhen/docs/memos/PROFESSOR_QUERY_RESIDUAL_V11.md`
- `legacy/shenzhen/docs/memos/PROFESSOR_QUERY_MACRO_LOSS.md`
- `legacy/shenzhen/docs/memos/PROFESSOR_UPDATE_BATCH_EPE.md`
- `docs/archive/sota/SOTA_TRAJECTORY_GENERATION.md`

---

## 2) 分层约定（避免“文档—代码—结果”不一致）

### 2.1 协议层（必须严格遵循）

- `docs/TASK_DEFINITION.md`  
  - 任务定义（KnownDestination）、`vel` 语义（step displacement）、Phase A/B 边界  
  - dt-fixed（Phase B 必须 dt=30s）  
  - 无泄漏原则（train-only 产物合同）  
  - 评估协议（K、指标口径）
- `docs/DATA_CONTRACT.md`
  - Phase D 多源数据版本/时间/坐标系/road-type 口径（避免“数据对不齐导致指标漂移”）

### 2.2 事实层（只写“仓库里已有产物”）

- `legacy/shenzhen/docs/legacy_shenzhen/PHASE_A_RESULTS.md`（legacy）
- `legacy/shenzhen/docs/legacy_shenzhen/PHASE_C_RESULTS.md`（legacy）

写作规则：
- 只引用仓库内可点击路径（例如 `legacy/shenzhen/data/experiments/.../metrics.json`）
- 每条结论都标注是否 *preliminary*（quick / subset / 单 seed）

### 2.3 诊断与方案层（允许假设，但要可证伪）

- `legacy/shenzhen/docs/legacy_shenzhen/HIERARCHICAL_VALIDATION_PROTOCOL.md`：分层验证协议（legacy，G1/G2 审计命令）
- `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`：新路线图（OSM/拓扑/语义/Diffusion）
- `docs/SOTA_TRAJECTORY_GENERATION_2025_UPDATE.md`：学术 SOTA（含 2025 更新）

### 2.4 写作/汇报层（不作为真相源）

- `legacy/shenzhen/docs/legacy_shenzhen/PPT_SPEAKER_NOTES.md`：逐页讲稿（legacy，服务于汇报，不替代结果文档）
- `docs/ESSAY_QUICK_GUIDE.md`：写作流程与素材入口

---

## 3) 常见踩坑（先读再跑）

- **跨机器跑实验**：先确认 `DATA/NAV/PRIOR/CKPT` 在目标机器都存在；缺一个就会 `FileNotFoundError` 直接退出。
- **tmux “没进度”**：不要用 `>log 2>&1 &` 隐藏；建议 `python -u ... |& tee logs/xxx.log`，另开窗口 `tail -f`。
- **HDF5 多进程/锁**：多进程 dataloader/并行评估可能卡锁；建议 `HDF5_USE_FILE_LOCKING=FALSE`，必要时 `--num_workers 0`。
- **长时间 eval**：先做 `K=1 + max_batches=50` 粗筛，再做 `K=10 + max_batches=200` 精验，最后才跑 full test。

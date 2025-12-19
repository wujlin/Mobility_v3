# docs 索引（单一真相源）

> 目的：把“规范 / 结果 / 诊断 / 方案 / 写作素材”分层，避免文档互相打架，避免重复踩坑。  
> 原则：**TASK_DEFINITION 是协议真相源**；其余文档要么是结果事实总结，要么是阶段性诊断与方案备忘。

---

## 1) 先看哪份？（按需求）

- **我要确认任务定义/无泄漏/评估协议**：`docs/TASK_DEFINITION.md`
- **我要快速理解代码框架与模块关系**：`docs/CODE_STRUCTURE.md`
- **我要理解数据产物与字段**：`docs/DATA_STRUCTURE.md`
- **我要写论文/essay 的 Phase A 素材**：`docs/PHASE_A_RESULTS.md`
- **我要写论文/essay 的 Phase B 素材（dt30 严格版）**：`docs/PHASE_B_RESULTS.md`
- **我要复盘“收缩/宏观偏小”的根因**：`docs/ROOT_CAUSE_ANALYSIS.md`
- **我要跑/复现 v1.1（Residual Diffusion）**：`docs/RESIDUAL_DIFFUSION.md`
- **我要做地理可视化（bbox 映射，不做 map-matching）**：`docs/GEO_VISUALIZATION.md`
- **我要给教授/专家咨询（问题包）**：`docs/EXPERT_CONSULTATION_PACKET.md`、`docs/PROFESSOR_QUERY_MACRO_LOSS.md`
- **我要准备汇报讲稿**：`docs/PPT_SPEAKER_NOTES.md`
- **我要少踩坑地跑实验/同步多机器**：`docs/EXPERIMENT_PLAYBOOK.md`

---

## 2) 分层约定（避免“文档—代码—结果”不一致）

### 2.1 协议层（必须严格遵循）

- `docs/TASK_DEFINITION.md`  
  - 任务定义（KnownDestination）、`vel` 语义（step displacement）、Phase A/B 边界  
  - dt-fixed（Phase B 必须 dt=30s）  
  - 无泄漏原则（train-only 产物合同）  
  - 评估协议（K、指标口径）

### 2.2 事实层（只写“仓库里已有产物”）

- `docs/PHASE_A_RESULTS.md`
- `docs/PHASE_B_RESULTS.md`

写作规则：
- 只引用仓库内可点击路径（例如 `data/experiments/.../metrics.json`）
- 每条结论都标注是否 *preliminary*（quick / subset / 单 seed）

### 2.3 诊断与方案层（允许假设，但要可证伪）

- `docs/ROOT_CAUSE_ANALYSIS.md`：问题机制与排雷证据链
- `docs/RESIDUAL_DIFFUSION.md`：v1.1 结构性修复（prior + residual）
- `docs/PHASE_B_REVIEW.md`：外部 review 记录与回应

### 2.4 写作/汇报层（不作为真相源）

- `docs/PPT_SPEAKER_NOTES.md`：逐页讲稿（服务于汇报，不替代结果文档）
- `docs/ESSAY_QUICK_GUIDE.md`：写作流程与素材入口

---

## 3) 常见踩坑（先读再跑）

- **跨机器跑实验**：先确认 `DATA/NAV/PRIOR/CKPT` 在目标机器都存在；缺一个就会 `FileNotFoundError` 直接退出。
- **tmux “没进度”**：不要用 `>log 2>&1 &` 隐藏；建议 `python -u ... |& tee logs/xxx.log`，另开窗口 `tail -f`。
- **HDF5 多进程/锁**：多进程 dataloader/并行评估可能卡锁；优先 `HDF5_USE_FILE_LOCKING=FALSE`，必要时 `--num_workers 0`。
- **长时间 eval**：先做 `K=1 + max_batches=50` 粗筛，再做 `K=10 + max_batches=200` 精验，最后才跑 full test。

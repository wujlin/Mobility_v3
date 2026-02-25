# Docs 审计清单（2026-02-25）

目标：识别 `docs/` 中与当前 Porto + WayCASD 主线不一致或冗余的文档，减少仓库噪音。

## 执行状态（2026-02-25）

- 已完成一次结构化整理：`ARCHIVE` 集合已迁移到 `docs/archive/legacy_20260225/`。
- `docs/README.md` 已重写为“当前主线 + 归档入口”。
- 活跃文档中的旧路径引用已修复到新路径（或归档路径）。

判定标签：
- `KEEP`：当前主线直接使用。
- `UPDATE`：有价值但口径混杂，需瘦身/改名/迁移。
- `ARCHIVE`：历史方向材料，建议迁入 `docs/archive/`。
- `ARCHIVED`：已在 archive。

## 1) KEEP（当前主线）
- `docs/ICML_2026_WAYCASD_EXPERIMENT_LOG.md`
- `docs/WAYCASD_METHOD_COMMITMENT.md`
- `docs/WAYCASD_FINDINGS_COMMITMENT.md`
- `docs/DATA_MIGRATION_PORTO.md`
- `docs/WAY_CASD_ARCHITECTURE.md`
- `docs/baseline_sota_exp.md`
- `docs/corridor_analysis.md`
- `docs/Literature_review/Literature_review_7.md`
- `docs/Literature_review/Literature_review_exposureBias.md`
- `docs/Literature_review/Literature_review_segment.md`

## 2) UPDATE（保留但需要整理）
- `docs/CODE_STRUCTURE.md`（混有 RouteGen/Paper2/legacy 三套叙事）
- `docs/DATA_STRUCTURE.md`（Detroit/WorldTrace 段落过多，Porto 口径应置顶）
- `docs/PI_BRIEF_ROUTEGEN_ICML2026.md`（2026-01 版本，实验结论已滞后）
- `docs/TASK_DEFINITION.md`（任务定义过宽，需拆主线与历史线）
- `docs/WORKSTATION_GUIDE.md`（可运行但历史说明过长）
- `docs/CORRIDOR_EXP_PROMPT.md`（建议移到 `docs/prompts/`）
- `docs/WAY_CASD_FIGURE_PROMPT.md`（建议移到 `docs/prompts/`）
- `docs/WAY_CASD_RESULTS_NARRATIVE.md`（与 Findings 有重叠，建议合并）
- `docs/CORRIDOR_DIVERSITY_LITERATURE_V2.md`（文献可保留，建议统一命名）

## 3) ARCHIVE（当前主线已过期/非主线）
- `docs/CORRIDOR_DIVERSITY_LITERATURE.md`
- `docs/DATA_CONTRACT.md`
- `docs/DETROIT_UNDER_OVERUSE_FIELD_STATUS.md`
- `docs/ESSAY_QUICK_GUIDE.md`
- `docs/ICML_2026_EXPERIMENT_PLAN_ROUTEGEN.md`
- `docs/ICML_2026_ROUTEGEN_SYNC_MANIFEST.md`
- `docs/ICML_2026_ROUTEGEN_SYNC_MANIFEST.json`
- `docs/Literature_review/Literature_review_avoidence.md`
- `docs/New_story.md`
- `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`
- `docs/RESEARCH_LOG.md`
- `docs/WAYBACK.md`
- `docs/WAY_CASD_METHOD.md`
- `docs/WORDTRACE_UNITRAJ.md`
- `docs/WORLDTRACE_OWNER_AUDIT.md`
- `docs/WORLDTRACE_SPATIAL_VIZ.md`
- `docs/visual_style_guide.md`

## 4) ARCHIVED（已归档）
- `docs/archive/sota/SOTA_TRAJECTORY_GENERATION.md`

## 5) 发现的结构异常
- Git 状态显示：`docs/README.md` 被删除，仓库根目录出现未跟踪 `README.md`。
- 该异常需先确认意图（是要把 docs 索引上移到仓库根，还是误操作）。

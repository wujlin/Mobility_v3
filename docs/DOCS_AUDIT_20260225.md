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
- `docs/WAYCASD_EXPERIMENT_LOG.md`
- `docs/WAYCASD_METHOD_COMMITMENT.md`
- `docs/WAYCASD_FINDINGS_COMMITMENT.md`
- `docs/DATA_MIGRATION_PORTO.md`
- `docs/Literature_review/Literature_review_7.md`
- `docs/Literature_review/Literature_review_exposureBias.md`
- `docs/Literature_review/Literature_review_segment.md`

## 2) UPDATE（保留但需要整理）
- `docs/CODE_STRUCTURE.md`（混有 RouteGen/Paper2/legacy 三套叙事）
- `docs/DATA_STRUCTURE.md`（Detroit/WorldTrace 段落过多，Porto 口径应置顶）
- `docs/TASK_DEFINITION.md`（任务定义过宽，需拆主线与历史线）
- `docs/WORKSTATION_GUIDE.md`（可运行但历史说明过长）
- `docs/CORRIDOR_DIVERSITY_LITERATURE_V2.md`（文献可保留，建议统一命名）

## 3) ARCHIVED（已归档，历史非主线）
- `docs/archive/sota/SOTA_TRAJECTORY_GENERATION.md`
- `docs/archive/legacy_20260225/CORRIDOR_DIVERSITY_LITERATURE.md`
- `docs/archive/legacy_20260225/DATA_CONTRACT.md`
- `docs/archive/legacy_20260225/DETROIT_UNDER_OVERUSE_FIELD_STATUS.md`
- `docs/archive/legacy_20260225/ESSAY_QUICK_GUIDE.md`
- `docs/archive/legacy_20260225/ICML_2026_EXPERIMENT_PLAN_ROUTEGEN.md`
- `docs/archive/legacy_20260225/ICML_2026_ROUTEGEN_SYNC_MANIFEST.md`
- `docs/archive/legacy_20260225/ICML_2026_ROUTEGEN_SYNC_MANIFEST.json`
- `docs/archive/legacy_20260225/Literature_review/Literature_review_avoidence.md`
- `docs/archive/legacy_20260225/New_story.md`
- `docs/archive/legacy_20260225/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`
- `docs/archive/legacy_20260225/RESEARCH_LOG.md`
- `docs/archive/legacy_20260225/WAYBACK.md`
- `docs/archive/legacy_20260225/WAY_CASD_METHOD.md`
- `docs/archive/legacy_20260225/WAY_CASD_ARCHITECTURE.md`
- `docs/archive/legacy_20260225/WAY_CASD_RESULTS_NARRATIVE.md`
- `docs/archive/legacy_20260225/WAY_CASD_FIGURE_PROMPT.md`
- `docs/archive/legacy_20260225/CORRIDOR_EXP_PROMPT.md`
- `docs/archive/legacy_20260225/PI_BRIEF_ROUTEGEN_ICML2026.md`
- `docs/archive/legacy_20260225/baseline_sota_exp.md`
- `docs/archive/legacy_20260225/corridor_analysis.md`
- `docs/archive/legacy_20260225/WORDTRACE_UNITRAJ.md`
- `docs/archive/legacy_20260225/WORLDTRACE_OWNER_AUDIT.md`
- `docs/archive/legacy_20260225/WORLDTRACE_SPATIAL_VIZ.md`
- `docs/archive/legacy_20260225/visual_style_guide.md`

## 5) 发现的结构异常
- 当前无阻塞异常（主索引与归档路径已对齐）。

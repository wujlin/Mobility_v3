# legacy_mapfree（封存：map-free 线路）

本目录用于**封存**此前的 map-free / grid-only 叙事与实现，避免后续 map-aware 主线开发时反复改坏旧链路或丢失证据。

## 为什么要封存（问题 -> 结论）

核心结论：此前 map-free 线路在“route generation（trip-level 路线选择）”这个问题上出现了**任务定义与数据处理不一致**的问题——滑窗 `F=256` 会把完整 trip 切成短片段，导致“走廊选择/多模态”在数据里不可见，从而让语义条件化与 corridor-level 指标失去意义。

更完整的技术诊断见 `legacy_mapfree/conclusions.md`。

## 这里保留什么

- **可复用组件（建议在新主线继续沿用）**
  - 时间特征：`src/features/temporal.py::encode_route_temporal_2d`
  - 执行阶段 diffusion（轨迹细化）：`src/training/train_route_exec_diffusion_wp_npz.py` 及其采样/评估脚本
  - 评估与可视化脚本：`src/evaluation/*`、`src/plot_style.py`

- **map-free 主线关键证据（实验产物不入 git）**
  - 本地同步目录口径：`_sync/wsa/icml2026_routegen/<EXP_DIR>/`
  - 关键 gate 的 JSON/PDF/PNG：参见 `legacy_mapfree/experiments/README.md`

## 如何复现旧结果（只给口径，不拉大文件进仓库）

1. 在 wsA 上运行对应实验，产物落到：
   - `$RAW_ROOT/experiments/icml2026_routegen/<EXP_DIR>/`
2. rsync 到本地：
   - `rsync -avP wsA:"$RAW_ROOT/experiments/icml2026_routegen/<EXP_DIR>/" "_sync/wsa/icml2026_routegen/<EXP_DIR>/"`

> 说明：旧线路的“可复现”依赖于你们当时的实验目录与 rsync 落盘口径，因此这里不做大规模复制/搬运（避免不可逆的目录漂移）。

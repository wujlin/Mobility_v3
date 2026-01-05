## Legacy（深圳 / dt30）封存区

本目录用于**封存**深圳出租车（dt30）相关的旧数据、旧脚本与旧文档，避免与当前主线（WorldTrace × Detroit / Phase D）在查阅与检索时互相污染。

### 目录结构（按用途分层）

- `legacy/shenzhen/docs/`
  - `legacy/shenzhen/docs/legacy_shenzhen/`：Phase C（trip-level）历史证据链与审计协议
  - `legacy/shenzhen/docs/phase_b/`：Phase B（窗口级 / dt30）材料
  - `legacy/shenzhen/docs/memos/`：阶段性备忘与外部咨询记录
- `legacy/shenzhen/data/`：旧流水线产物（`processed_dt30/`、`processed_passenger_dt30/`、`experiments/`、`raw/` 等）
- `legacy/shenzhen/scripts/`：旧脚本（Phase B 训练链等）
- `legacy/shenzhen/geo_map/`：深圳底图（区县边界等）
- `legacy/shenzhen/slide/`：旧汇报素材
- `legacy/shenzhen/track_data/`：原始深圳 GPS 数据（不入 git）
- `legacy/shenzhen/logs/`、`legacy/shenzhen/old_logs/`：旧日志（不入 git）

### 使用约定

- 当前主线文档只保留**最小入口引用**：需要回溯深圳/ dt30 时从本目录进入，不在主线文档中展开细节。
- 检索时建议显式排除：`rg ... --glob '!legacy/shenzhen/**'`。

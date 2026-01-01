# Wayback 遥感（ArcGIS Wayback）下载与落盘约定

> 目标：在给定 bbox + zoom 下，下载 ArcGIS Wayback 的多期影像瓦片（**metadata-first**：只下载“发生过变化”的 release）。  
> 口径：Detroit 主线 bbox/grid 以 `docs/DATA_CONTRACT.md` 为准（Detroit core：`[-83.25, 42.25, -82.95, 42.50]`，`1024×1024`）。

本仓库已提供可直接运行的脚本：
- `src/data/wayback/download_wayback_tiles.py`

---

## 1) 输出目录结构

执行一次下载会在 `--out_dir` 下生成：
- `wayback_scan_meta.json`：扫描 bbox/zoom 后的元数据统计（tile_range、扫描 tile 数、下载任务数等）
- `wayback_download_report.json`：下载完成后的统计（OK/SKIP/FAIL 与耗时）
- 瓦片目录（按空间瓦片分组、按 release_date 命名）：
  - `z{zoom}/{zoom}_{x}_{y}/{release_date}.jpg`

---

## 2) 快速 Smoke Test（建议先做）

在仓库根目录执行（工作站 A 或本地均可）：

```bash
export DATA_ROOT="$HOME/data/mobility_data"
mkdir -p "$DATA_ROOT/wayback"

# 先 dry-run：只扫描元数据，不下载
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$DATA_ROOT/wayback/detroit_smoke_z16_dryrun" \
  --bbox -83.06 42.32 -83.03 42.34 \
  --zoom 16 \
  --max_threads 16 \
  --max_tiles 200 \
  --dry_run

# 再小规模真实下载（仍限制 max_tiles）
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$DATA_ROOT/wayback/detroit_smoke_z16" \
  --bbox -83.06 42.32 -83.03 42.34 \
  --zoom 16 \
  --max_threads 16 \
  --max_tiles 200
```

---

## 3) Detroit core bbox（全量范围；建议先 dry-run 看任务量）

```bash
export DATA_ROOT="$HOME/data/mobility_data"
mkdir -p "$DATA_ROOT/wayback"

python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$DATA_ROOT/wayback/detroit_core_z16_dryrun" \
  --bbox -83.25 42.25 -82.95 42.50 \
  --zoom 16 \
  --max_threads 16 \
  --dry_run
```

如果 `download_tasks` 数量可接受，再去掉 `--dry_run` 开始下载：

```bash
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$DATA_ROOT/wayback/detroit_core_z16" \
  --bbox -83.25 42.25 -82.95 42.50 \
  --zoom 16 \
  --max_threads 16
```

---

## 4) 常见问题

### 4.0 `num_releases_total=0` 或 `download_tasks=0`（但 bbox/zoom 正常）

这是“Wayback 接口变化”的典型症状（不是缓存/不是反爬）：
- **config schema 变了**：`waybackconfig.json` 不再包含旧版的 `archive` 字段；
- **metadata endpoint 变了**：旧的 `wayback-tilemap-console` bucket 可能已经下线。

脚本现在支持通过参数覆盖：
- `--config_url`
- `--metadata_url_tpl`

### 4.1 `ModuleNotFoundError: No module named 'src.data.wayback'`

这不是 Python 结构问题，而是目标机器的仓库版本里缺少 `src/data/wayback/` 目录。  
解决方式：把本地仓库的 `src/data/wayback/` 同步到工作站 A 的同一 repo 下（建议用 `rsync`）。

### 4.2 任务量太大/下载太慢

建议：
- 先用 `--dry_run` 估算任务量；
- 先降低 `--zoom` 或缩小 `--bbox`；
- 用 `--max_tiles` 做分块下载（每块输出到独立 `out_dir`，方便断点续跑与审计）。

### 4.3 metadata 端点下线（`NoSuchBucket`）

如果运行报错包含 `NoSuchBucket`，说明默认的 metadata URL 已失效，需要你先通过浏览器/抓包确认最新端点，再用：

```bash
python -m src.data.wayback.download_wayback_tiles \
  --out_dir ... \
  --bbox ... \
  --zoom 16 \
  --metadata_url_tpl "https://<NEW_ENDPOINT>/metadata/edge/tile/{z}/{y}/{x}.json"
```

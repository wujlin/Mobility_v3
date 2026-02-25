# Wayback 遥感（ArcGIS Wayback）下载与落盘约定

> 目标：在给定 bbox + zoom 下，下载 ArcGIS Wayback 的多期影像瓦片。  
> 现实约束：当前观测到 Wayback 的 **metadata bucket 已下线**（`NoSuchBucket`），因此默认的 **metadata-first** 模式可能不可用；脚本已提供 `fixed_releases` 降级模式（按指定 release_ids 直接下载）。  
> 口径：Detroit 主线 bbox/grid 以 `docs/DATA_CONTRACT.md` 为准（Detroit core：`[-83.25, 42.25, -82.95, 42.50]`，`1024×1024`）。

本仓库已提供可直接运行的脚本：
- `src/data/wayback/download_wayback_tiles.py`

---

## 1) 输出目录结构

执行一次下载会在 `--out_dir` 下生成：
- `wayback_scan_meta.json`：扫描 bbox/zoom 后的元数据统计（tile_range、扫描 tile 数、下载任务数等）
- `wayback_download_report.json`：下载完成后的统计（OK/SKIP/FAIL 与耗时）
- 瓦片目录（按空间瓦片分组、按 release_id 命名；避免 release_date 缺失/异常导致口径不一致）：
  - `z{zoom}/{zoom}_{x}_{y}/rid_{release_id}.jpg`

---

## 2) 快速 Smoke Test（建议先做）

在仓库根目录执行（工作站 A 或本地均可）：

```bash
export RAW_ROOT="$HOME/data/geoexplicit_data"
mkdir -p "$RAW_ROOT/wayback"

# Step 0：列出 release 列表（不依赖 metadata）
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$RAW_ROOT/wayback/_list_releases" \
  --list_releases 20

# Step 1：metadata-first（若 metadata 端点仍可用，则可只下载“发生过变化”的 release）
# 先 dry-run：只扫描元数据，不下载
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$RAW_ROOT/wayback/detroit_smoke_z16_dryrun" \
  --bbox -83.06 42.32 -83.03 42.34 \
  --zoom 16 \
  --max_threads 16 \
  --max_tiles 200 \
  --dry_run

# Step 2：若上一步报 `NoSuchBucket`（metadata 端点下线），改用 fixed_releases：
# 1) 先从 Step 0 的输出中挑选若干 release_id（建议 3~6 个）
# 2) 把它们填到 --release_ids 后面（示例中的数字仅作演示）
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$RAW_ROOT/wayback/detroit_smoke_z16_fixed" \
  --bbox -83.06 42.32 -83.03 42.34 \
  --zoom 16 \
  --max_threads 16 \
  --max_tiles 200 \
  --mode fixed_releases \
  --release_ids 10312 11019 1296
```

如果你希望完全不碰 metadata-first（因为它现在不可用），可以直接跳到 Step 2。

---

## 3) Detroit core bbox（全量范围；建议先 dry-run 看任务量）

```bash
export RAW_ROOT="$HOME/data/geoexplicit_data"
mkdir -p "$RAW_ROOT/wayback"

python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$RAW_ROOT/wayback/detroit_core_z16_dryrun" \
  --bbox -83.25 42.25 -82.95 42.50 \
  --zoom 16 \
  --max_threads 16 \
  --dry_run
```

如果你确认 metadata-first 可用且 `download_tasks` 数量可接受，再去掉 `--dry_run` 开始下载：

```bash
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$RAW_ROOT/wayback/detroit_core_z16" \
  --bbox -83.25 42.25 -82.95 42.50 \
  --zoom 16 \
  --max_threads 16
```

如果 metadata-first 不可用，使用 fixed_releases（强烈建议加 `--max_tiles` 分块下载）：

```bash
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$RAW_ROOT/wayback/detroit_core_z16_fixed_part01" \
  --bbox -83.25 42.25 -82.95 42.50 \
  --zoom 16 \
  --max_threads 16 \
  --max_tiles 2000 \
  --mode fixed_releases \
  --release_ids 10312 11019 1296
```

---

## 4) 常见问题

### 4.0 `num_releases_total=0` 或 `download_tasks=0`（但 bbox/zoom 正常）

这是“Wayback 接口变化”的典型症状（不是缓存/不是反爬）：
- **config schema 变了**：`waybackconfig.json` 不再包含旧版的 `archive` 字段；
- **metadata endpoint 变了**：旧的 `wayback-tilemap-console` bucket 可能已经下线。

脚本现在支持：
- `--config_url`
- `--metadata_url_tpl`
- `--list_releases`
- `--mode fixed_releases --release_ids ...`（在 metadata 端点不可用时用于降级）

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
  --out_dir "$RAW_ROOT/wayback/detroit_smoke_z16_dryrun_newmeta" \
  --bbox -83.06 42.32 -83.03 42.34 \
  --zoom 16 \
  --metadata_url_tpl "https://<NEW_ENDPOINT>/metadata/edge/tile/{z}/{y}/{x}.json"
```

在你还没拿到新 metadata 端点之前，建议直接使用 `fixed_releases` 跑通数据落盘（见 2/3 节）。

### 4.4 `SSLError: CERTIFICATE_VERIFY_FAILED (Hostname mismatch)`

在部分网络环境下（尤其存在透明代理/证书注入时），`requests` 可能会报主机名不匹配的 SSL 错误，导致下载任务“看起来在跑但没有任何文件落盘”。

**排查与解决（KISS）**：
- 先用浏览器或 `curl` 验证同一 URL 是否能正常返回 `image/jpeg`。
- 若你有本地代理（例如 Clash），直接在运行前显式导出：

```bash
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
```

然后重新运行下载命令。

> 说明：我们不在脚本里默认关闭 SSL 校验（避免把“下载成功”建立在不安全假设上）；网络侧问题应优先用代理/CA 修复。

### 4.5 `wayback_download_report.json` 为空/无法解析

`download_wayback_tiles.py` 会在 `--out_dir` 内 **自动写入**：
- `wayback_scan_meta.json`
- `wayback_download_report.json`

因此不要把 stdout 重定向到 `wayback_download_report.json`（会生成一个空文件覆盖真实报告）。

如果你想同时保留 CLI 输出与 stderr，建议：

```bash
PYTHONUNBUFFERED=1 python -m src.data.wayback.download_wayback_tiles ... \
  > >(tee "$OUT_DIR/cli_stdout.json") \
  2> >(tee "$OUT_DIR/cli_stderr.log" >&2)
```

# 工作站/本地环境使用说明（运行与数据落盘口径）

> 目标：把“在哪跑、用哪个环境、数据落哪里、日志怎么留、常见故障怎么定位”写成一份可复现口径，避免跨机器/跨人协作时反复踩坑。  
> 说明：**数据契约**以 `docs/DATA_CONTRACT.md` 为准；**数据目录结构**以 `docs/DATA_STRUCTURE.md` 为准；本文只记录运行环境与操作习惯的约定。

---

## 0) 一句话约定（最重要）

- 仓库只放代码与小体量产物；大数据一律落到外置目录 `$RAW_ROOT`（不进 git），用软链/环境变量引用。

---

## 1) 机器与路径约定

### 1.1 工作站 A（训练/大规模处理）

- **仓库路径**：`~/projects/Mobility_v3`
- **外置数据根目录（唯一口径）**：`$RAW_ROOT=/home/jinlin/data/geoexplicit_data`
- **常用数据目录**（示例）：
  - WorldTrace：`$RAW_ROOT/worldtrace/`
  - OSM：`$RAW_ROOT/osm/`
  - SafeGraph：`$RAW_ROOT/safegraph/`
  - Wayback：`$RAW_ROOT/wayback/`
  - Census：`$RAW_ROOT/census/`

建议把这两个变量写进 `~/.bashrc`（只做本机约定，不要写进仓库脚本）：

```bash
export REPO="$HOME/projects/Mobility_v3"
export RAW_ROOT="$HOME/data/geoexplicit_data"
```

#### 1.1.1 工作站 A 数据内容（当前主线：WorldTrace Detroit/Columbus）

> 口径：以下路径是当前主线（Detroit story）在工作站 A 上的**实际落盘结构**；不涉及 legacy 深圳分析。

| 数据 | 目录 | 关键文件（示例） | 说明 |
|---|---|---|---|
| WorldTrace 原始包 | `$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/` | `Trajectory.zip`, `Meta.zip` | 主数据底座（大文件，不进 git） |
| Detroit segments | `$RAW_ROOT/worldtrace/detroit_core_v1/` | `segments.parquet` | Detroit core 连续段（建模/审计输入） |
| Columbus segments | `$RAW_ROOT/worldtrace/columbus_core_v1/` | `segments.parquet` | 参考城市（构造 behavioral reference） |
| OSM（Detroit） | `$RAW_ROOT/osm/` | `michigan-latest.osm.pbf` | 用于生成 `road_prob/dist_to_road`（soft prior） |
| OSM（Columbus） | `$RAW_ROOT/osm/` | `ohio-latest.osm.pbf` | 同上 |
| OSM 软先验产物（每城） | `$RAW_ROOT/worldtrace/<city>_core_v1/` | `osm_road_prob.npy`, `osm_dist_to_road_m.npy`, `osm_road_prob_meta.json` | **唯一口径以 meta 为准**（variant/sigma/buffer/tier_weights 等） |
| SafeGraph（POI shards） | `$RAW_ROOT/safegraph/safegraph_unzip/` | `Global_Places_POI_Data-*.csv` | 目前为 Places Base 分片；Rich/Geometry 若缺失需在契约里标注 |
| POI 栅格化产物（每城） | `$RAW_ROOT/worldtrace/<city>_core_v1/` | `poi_density_*.npy`, `poi_raster_meta.json` | 必须写清 `grid_shape`/bbox 与过滤口径 |
| Wayback 遥感（Detroit） | `$RAW_ROOT/wayback/detroit_core_z16_fixed_multi_r6/` | `wayback_scan_meta.json`, `z16/.../rid_<release_id>.jpg` | 以 `release_id` 作为快照标识；不以 release_date 做时间证据 |
| Census/ACS（Detroit） | `$RAW_ROOT/census/detroit_core_v1/` | `acs_tract_*.csv`, `tract_covariates_*.parquet` | TIGER 若遇 403 可手动下载；注意 GeoParquet vs 普通 parquet 读取方式 |

### 1.2 本地 WSL（写作/轻量分析/拉图）

- **仓库路径（WSL）**：`/mnt/e/newdesktop/HKUST/GeoExplicit_SFM/v3`
- 若需要从工作站拉取图/JSON：使用 `rsync -avP wsA:... local/...`（见第 6 节）

---

## 2) Python/环境口径

仓库代码默认按 `python -m src...` 运行，要求：

- 当前工作目录在仓库根（即 `pwd == $REPO`）
- `python` 对应的环境里安装了 `requirements.txt` 所需依赖

建议实践（不是硬要求）：

- 训练/评估：用一个统一 conda env（例如 `dpl`）
- 地理处理（OSM/Wayback/Census）：可以单独一个 env（例如 `geo`），但**避免同一任务跨 env 混跑**

快速自检（跑任何长任务前）：

```bash
python -c "import sys; print(sys.executable)"
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

---

## 3) 日志与可观测性（避免“跑了但不知道在干嘛”）

### 3.1 推荐的运行方式

所有长任务都用**非缓冲输出**并落盘日志：

```bash
PYTHONUNBUFFERED=1 python -u -m <module> <args...> |& tee "<out_dir>/run.log"
```

另开窗口看进度：

```bash
tail -f "<out_dir>/run.log"
```

> 不要用 `>log 2>&1 &` 把输出藏起来；这会让“卡住/报错/被杀”无法定位。

### 3.2 `set -euo pipefail` 是什么？为什么看起来像“窗口断开”？

它是 bash 的“快速失败”开关：

- `-e`：任一命令失败（非 0）就立刻退出
- `-u`：引用未定义变量就报错退出
- `pipefail`：管道里任一环节失败都算失败

**它不会主动断开 ssh/tmux**；但如果脚本中间有一条命令失败，整个脚本会立即结束，ssh 远程命令结束后你会回到本地 shell，看起来像“刚跑完/突然没了”。因此：

- 交互式排错阶段不建议加这句
- 批处理脚本阶段可以加，但要确保日志完整落盘

---

## 4) tmux 使用约定（工作站 A 必备）

典型流程：

```bash
tmux new -s detroit
# 里面跑任务…
# Ctrl-b d 退出
tmux attach -t detroit
```

建议：

- 每个长任务输出到独立 `out_dir`，不要共享一个 `run.log`
- tmux 里跑多进程任务时，优先保证日志可追踪（见第 3 节）

---

## 5) 多进程/并发（常见坑与可复现实践）

### 5.1 ProcessPool 的两类典型错误

- `BrokenProcessPool`：子进程被异常杀死（内存/句柄/依赖/数据损坏）
- `OSError: handle is closed`：常见于 `spawn` 启动方式下的句柄序列化/关闭时序问题

### 5.2 建议的跑法（以 WorldTrace segments 构建为例）

在 Linux 上默认优先用 `fork`；若确有需要再切 `spawn`：

```bash
python -m src.data.worldtrace.build_detroit_segments \
  --trajectory_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Trajectory.zip" \
  --out_parquet "$RAW_ROOT/worldtrace/<city>_core_v1/segments.parquet" \
  --num_workers 24 \
  --chunk_size 5000 \
  --mp_start fork \
  ...
```

如果出现不稳定（随机崩/句柄错误），只调整两件事（保持 KISS）：

- 降低 `--num_workers`
- 增大 `--chunk_size`（减少进程间通信频率）

---

## 6) 跨机器传输（图/JSON/小产物）

从工作站 A 拉图到本地（WSL）示例：

```bash
rsync -avP wsA:"$RAW_ROOT/worldtrace/detroit_core_v1/story/" \
  "/mnt/e/newdesktop/HKUST/GeoExplicit_SFM/v3/essay/figures/worldtrace_detroit/story/"
```

建议：

- 只拉 `png/pdf/json` 等小文件，不要拉 zip/pbf 大文件到本地
- 若网络不稳：加 `--partial --append-verify` 断点续传

---

## 7) 代理与网络（Wayback / Census 常见问题）

### 7.1 Wayback（ArcGIS）SSL hostname mismatch

现象：不挂代理时访问 `https://wayback.maptiles.arcgis.com/...` 可能出现证书 hostname mismatch。  
实践口径：**在运行 Wayback 下载前显式设置代理**（以 Clash 为例）：

```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```

可用性探测（最小 smoke test：只扫一个小 bbox/少量 tile；确认能下载到 jpg）：

```bash
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$RAW_ROOT/wayback/_probe_z16_r4756" \
  --bbox -83.06 42.32 -83.03 42.34 \
  --zoom 16 --max_threads 4 --max_tiles 10 \
  --mode fixed_releases --release_ids 4756
find "$RAW_ROOT/wayback/_probe_z16_r4756" -name "*.jpg" | wc -l
```

> Wayback 的下载/落盘约定详见 `docs/WAYBACK.md`。

### 7.2 Census TIGER 403

部分环境/时段会遇到 `www2.census.gov` 的 `403 Forbidden`。建议：

- 保留脚本自动下载的 ACS 指标表（一般可用）
- TIGER 边界若自动下载失败，可手动下载 zip 并转换为 GeoParquet（手动流程只做一次，落到 `$RAW_ROOT/census/...` 即可）

---

## 8) 大文件与 git（避免 push 崩盘）

- `$RAW_ROOT/` 下的大文件（WorldTrace zip / OSM pbf / SafeGraph shards / Wayback tiles）**绝不进入 git**
- legacy 深圳原始包（rar/大目录）也不要进 git；若误加入会触发 GitHub 大文件限制
- 仓库只提交：代码、文档、配置、小体量图表/JSON

---

## 10) 深圳数据封存（legacy，当前不参与分析）

当前主线是 **WorldTrace ×（Detroit/参考城市）**。仓库中的深圳相关目录仅作为历史复现材料封存：

- `legacy/shenzhen/`：保留旧实验与清洗口径的可复现入口

约束（必须遵守）：

- 任何 Detroit story 的图/表/结论 **不得引用深圳实验产物**（避免叙事口径混用）
- 新的分析脚本默认只读 `$RAW_ROOT/worldtrace/...`，不要把路径写回 `legacy/shenzhen/`

---

## 9) 与现有文档的关系（避免口径打架）

- 数据版本/坐标/时间：`docs/DATA_CONTRACT.md`
- 数据目录与字段：`docs/DATA_STRUCTURE.md`
- 当前主线路线图：`docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`
- Wayback 下载细节：`docs/WAYBACK.md`
